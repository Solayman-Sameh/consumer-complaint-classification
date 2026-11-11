"""Explicit PyTorch training loops; compatible completed runs are reusable."""
import gc
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import shutil
import time
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import collate_encoded, read_json, sha256, write_json
from .evaluation import metrics_from_predictions, save_evaluation, save_learning_curves
from .models import create_model, forward_batch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Repeated runs can still differ across hardware/CUDA/library versions.


def environment_record():
    versions = {}
    for name in ["torch", "transformers", "tokenizers", "numpy", "pandas", "scikit-learn", "safetensors"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "versions": versions,
            "cuda_version": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpus": [{"index": i, "name": torch.cuda.get_device_name(i),
                      "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                     for i in range(torch.cuda.device_count())]}


def experiment_signature(data, config):
    code = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        code.update(path.name.encode())
        code.update(path.read_bytes())
    payload = {"preparation_id": data.manifest["preparation_id"], "config": config,
               "code_sha256": code.hexdigest(), "versions": environment_record()["versions"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def make_loader(data, split, kind, config, device, indices=None, shuffle=False):
    batch_size = (config["transformer_batch_size"] if kind == "distilbert" else config["recurrent_batch_size"]) if shuffle else config["eval_batch_size"]
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(data.dataset(split, kind, indices), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_encoded, num_workers=config["num_workers"],
                      pin_memory=device.type == "cuda", generator=generator)


@torch.inference_mode()
def predict_loader(model, kind, loader, device, weights):
    model.eval()
    probabilities, positions = [], []
    total_loss, total_weight = 0.0, 0.0
    for batch in loader:
        targets = batch["labels"].to(device)
        logits = forward_batch(model, kind, batch, device).float()
        losses = nn.functional.cross_entropy(logits, targets, weight=weights, reduction="sum")
        total_loss += float(losses.item())
        total_weight += float(weights[targets].sum().item())
        probabilities.append(logits.softmax(dim=-1).cpu().numpy())
        positions.append(batch["indices"].numpy())
    return np.concatenate(probabilities), np.concatenate(positions), total_loss / total_weight


def load_checkpoint(kind, directory, data, config, device):
    checkpoint = torch.load(Path(directory) / "best.pt", map_location="cpu", weights_only=True)
    model = create_model(kind, data, config, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device)


@torch.inference_mode()
def benchmark_forward(model, kind, data, config, device):
    """Common validation examples, batch 1; precomputed IDs, transfer+forward only."""
    candidates = data.indices("validation")[:8]
    dataset = data.dataset("validation", kind, candidates)
    batches = [collate_encoded([dataset[i]]) for i in range(len(dataset))]
    model.eval()
    for batch in batches[:3]:
        forward_batch(model, kind, batch, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings = []
    for repeat in range(config["benchmark_repeats"]):
        start = time.perf_counter()
        forward_batch(model, kind, batches[repeat % len(batches)], device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - start) * 1000)
    return {"device": str(device), "batch_size": 1, "median_ms": float(np.median(timings)),
            "p95_ms": float(np.percentile(timings, 95)), "warmup_calls": min(3, len(batches)),
            "measured_calls": config["benchmark_repeats"], "includes_host_to_device_transfer": True,
            "includes_tokenization": False, "inputs": "Same first eight validation records; each model's prepared tokenization",
            "recurrent_max_length": config["recurrent_max_length"], "transformer_max_length": config["transformer_max_length"]}


def run_model(kind, data, config, output_root, device, verification=False, train_indices=None, validation_indices=None):
    if kind not in {"rnn", "lstm", "gru", "distilbert"}:
        raise ValueError(kind)
    device = torch.device(device)
    signature, signature_payload = experiment_signature(data, config)
    output = Path(output_root) / "models" / kind
    completed = output / "result.json"
    if completed.exists():
        result = read_json(completed)
        if result["signature"] != signature or result["verification"] != verification:
            raise RuntimeError(f"{kind}: existing outputs belong to a different experiment. Choose a new OUTPUT_DIR; existing results will not be overwritten.")
        if sha256(output / "best.pt") != result["checkpoint_sha256"]:
            raise RuntimeError(f"{kind}: checkpoint checksum failed")
        print(f"{kind}: reusing compatible completed run", flush=True)
        return result
    if output.exists() and any(output.iterdir()):
        previous = read_json(output / "run_identity.json") if (output / "run_identity.json").exists() else None
        if previous and previous["signature"] != signature:
            raise RuntimeError(f"{kind}: partial outputs have a different configuration. Choose a new OUTPUT_DIR.")
        print(f"{kind}: restarting an incomplete run from epoch 1 (weights-only checkpoints do not resume optimization).", flush=True)
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_identity.json", {"signature": signature, "verification": verification, "identity": signature_payload})
    seed_everything(config["seed"])
    start = time.perf_counter()
    model = create_model(kind, data, config, pretrained=True).to(device)
    loader = make_loader(data, "train", kind, config, device, train_indices, shuffle=True)
    validation = make_loader(data, "validation", kind, config, device, validation_indices)
    weights = torch.tensor(data.preprocessing["class_weights"], dtype=torch.float32, device=device)
    if kind == "distilbert":
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        groups = [{"params": [p for n, p in model.named_parameters() if not any(x in n for x in no_decay)], "weight_decay": config["transformer_weight_decay"]},
                  {"params": [p for n, p in model.named_parameters() if any(x in n for x in no_decay)], "weight_decay": 0.0}]
        optimizer = torch.optim.AdamW(groups, lr=config["transformer_lr"])
        from transformers import get_linear_schedule_with_warmup
        epochs = config["transformer_epochs"]
        accumulation = config["gradient_accumulation_steps"]
        updates = math.ceil(len(loader) / accumulation) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(updates * config["warmup_ratio"]), updates)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config["recurrent_lr"])
        epochs = config["recurrent_epochs"]
        accumulation = 1
        scheduler = None
    use_amp = config["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_f1, best_epoch, stale, monitor_best = -1.0, 0, 0, -1.0
    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    first_batches = []
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_weight = 0.0, 0.0
        for step, batch in enumerate(tqdm(loader, desc=f"{kind} epoch {epoch}/{epochs}", leave=False, disable=not sys.stderr.isatty())):
            batch_start = time.perf_counter()
            targets = batch["labels"].to(device, non_blocking=True)
            group_start = (step // accumulation) * accumulation
            group_size = min(accumulation, len(loader) - group_start)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = forward_batch(model, kind, batch, device)
                loss_sum = nn.functional.cross_entropy(logits.float(), targets, weight=weights, reduction="sum")
                weight_sum = weights[targets].sum()
                loss = loss_sum / weight_sum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{kind}: nonfinite training loss; no successful run is recorded")
            scaler.scale(loss / group_size).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None and scaler.get_scale() >= previous_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss_sum.detach().item())
            total_weight += float(weight_sum.item())
            if epoch == 1 and step < 20:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                first_batches.append(time.perf_counter() - batch_start)
                if step == 4:
                    seconds = float(np.mean(first_batches[1:]))
                    estimate = seconds * len(loader) * epochs / 60
                    print(f"\n{kind} pilot: {seconds:.3f} s/batch; about {estimate:.1f} min for {epochs} training epochs, excluding validation/downloads.", flush=True)
                    write_json(output / "pilot.json", {"seconds_per_batch_initial": seconds, "batches_per_epoch": len(loader),
                                                        "maximum_training_minutes_excluding_validation": estimate, "device": str(device)})
        probabilities, positions, validation_loss = predict_loader(model, kind, validation, device, weights)
        metrics = metrics_from_predictions(data.targets[positions], probabilities.argmax(axis=1), data.label_names)
        score = metrics["macro_f1"]
        history.append({"epoch": epoch, "training_loss": total_loss / total_weight, "validation_loss": validation_loss,
                        "validation_macro_f1": score, "validation_accuracy": metrics["accuracy"],
                        "epoch_seconds": time.perf_counter() - epoch_start, "learning_rate": optimizer.param_groups[0]["lr"]})
        write_json(output / "history.json", history)
        save_learning_curves(history, output)
        if score > best_f1:
            best_f1, best_epoch = score, epoch
            checkpoint = {"model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                          "kind": kind, "epoch": epoch, "validation_macro_f1": score, "signature": signature,
                          "verification": verification}
            temporary = output / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(output / "best.pt")
        if score > monitor_best + config["min_delta"]:
            monitor_best, stale = score, 0
        else:
            stale += 1
        print(f"{kind} epoch {epoch}: train loss {history[-1]['training_loss']:.4f}; val loss {validation_loss:.4f}; val macro F1 {score:.4f}", flush=True)
        if stale >= config["patience"]:
            break
    training_seconds = time.perf_counter() - start
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = load_checkpoint(kind, output, data, config, device)
    probabilities, positions, loss = predict_loader(model, kind, validation, device, weights)
    metrics = save_evaluation(output / "validation", probabilities, positions, data, loss, kind, "validation")
    if not math.isclose(metrics["macro_f1"], best_f1, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError("Reloaded checkpoint does not reproduce validation selection score")
    benchmark = benchmark_forward(model, kind, data, config, device)
    write_json(output / "inference_benchmark.json", benchmark)
    result = {"model": kind, "signature": signature, "verification": verification, "status": "complete",
              "best_epoch": best_epoch, "epochs_run": len(history), "training_seconds_including_initialization": training_seconds,
              "validation_macro_f1": metrics["macro_f1"], "validation_accuracy": metrics["accuracy"],
              "checkpoint_bytes": (output / "best.pt").stat().st_size, "checkpoint_sha256": sha256(output / "best.pt"),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "device": str(device), "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
              "batch_size": loader.batch_size, "gradient_accumulation_steps": accumulation,
              "amp": use_amp, "preparation_id": data.manifest["preparation_id"]}
    result["inference_batch1_median_ms"] = benchmark["median_ms"]
    write_json(completed, result)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
