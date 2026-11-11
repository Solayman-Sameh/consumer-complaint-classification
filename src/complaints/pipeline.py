"""Validation selection, one held-out test phase, export and bounded smoke checks."""
import gc
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
import pandas as pd
import torch

from .data import read_json, sha256, write_json
from .evaluation import save_evaluation
from .training import environment_record, experiment_signature, load_checkpoint, make_loader, predict_loader, run_model

REQUIRED_MODELS = ["rnn", "lstm", "gru", "distilbert"]


def freeze_selection(data, config, output_root):
    output_root = Path(output_root)
    signature, _ = experiment_signature(data, config)
    results = {}
    for kind in REQUIRED_MODELS:
        result = read_json(output_root / "models" / kind / "result.json")
        if result["verification"] or result["status"] != "complete" or result["signature"] != signature:
            raise RuntimeError("Final selection requires four compatible full-data runs")
        if sha256(output_root / "models" / kind / "best.pt") != result["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed after validation")
        results[kind] = result
    best_score = max(result["validation_macro_f1"] for result in results.values())
    tied = [kind for kind in config["tie_priority"] if results[kind]["validation_macro_f1"] >= best_score - config["tie_tolerance"]]
    winner = tied[0]
    selection = {"signature": signature, "selected_model": winner, "selection_metric": "validation_macro_f1",
                 "best_validation_macro_f1": best_score, "selected_validation_macro_f1": results[winner]["validation_macro_f1"],
                 "tie_tolerance": config["tie_tolerance"], "tie_priority": config["tie_priority"],
                 "rationale": "Highest validation macro F1; within 0.001 prefer the smaller architecture in the predeclared order.",
                 "checkpoint_sha256": {kind: result["checkpoint_sha256"] for kind, result in results.items()},
                 "test_outcomes_used_for_selection": False, "status": "frozen_before_test"}
    path = output_root / "selection.json"
    if path.exists() and read_json(path) != selection:
        raise RuntimeError("Frozen selection differs. Start a separately named experiment; test outcomes must not change the winner.")
    write_json(path, selection)
    rows = [{key: result[key] for key in ["model", "validation_macro_f1", "validation_accuracy", "best_epoch", "epochs_run", "training_seconds_including_initialization", "checkpoint_bytes", "trainable_parameters", "inference_batch1_median_ms"]}
            for result in results.values()]
    pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False).to_csv(output_root / "validation_comparison.csv", index=False)
    print("Frozen deployment selection:", winner, flush=True)
    return selection


def final_test(data, config, output_root, device):
    output_root = Path(output_root)
    selection = read_json(output_root / "selection.json")
    signature, _ = experiment_signature(data, config)
    if selection["signature"] != signature or selection["status"] != "frozen_before_test":
        raise RuntimeError("Freeze compatible validation selection before any test evaluation")
    completion = output_root / "final_test/complete.json"
    if completion.exists():
        saved = read_json(completion)
        if saved["signature"] != signature:
            raise RuntimeError("Existing test results have another experiment signature")
        print("Reusing completed final test phase; no new model selection.", flush=True)
        return saved
    weights = torch.tensor(data.preprocessing["class_weights"], dtype=torch.float32, device=device)
    results = {}
    for kind in REQUIRED_MODELS:
        if sha256(output_root / "models" / kind / "best.pt") != selection["checkpoint_sha256"][kind]:
            raise RuntimeError("Checkpoint changed since validation freeze")
        destination = output_root / "final_test" / kind
        marker = destination / "completed.json"
        if marker.exists():
            saved = read_json(marker)
            if saved["signature"] != signature:
                raise RuntimeError("Partial final-test results have another signature")
            results[kind] = read_json(destination / "metrics.json")
            continue
        model = load_checkpoint(kind, output_root / "models" / kind, data, config, device)
        loader = make_loader(data, "test", kind, config, device)
        probabilities, positions, loss = predict_loader(model, kind, loader, device, weights)
        results[kind] = save_evaluation(destination, probabilities, positions, data, loss, kind, "test")
        write_json(marker, {"signature": signature})
        del model
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    rows = [{"model": kind, **{key: metrics[key] for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1"]},
             "selected_on_validation": kind == selection["selected_model"]} for kind, metrics in results.items()]
    pd.DataFrame(rows).to_csv(output_root / "final_test/test_comparison.csv", index=False)
    summary = {"signature": signature, "selected_model": selection["selected_model"], "test_results": results,
               "policy": "Selection was frozen before this final test phase. Test outcomes do not change the deployment model."}
    write_json(completion, summary)
    return summary


def benchmark_bundle(predictor, texts, repeats):
    # CPU, batch size 1, preprocessing+tokenization+forward pass; warmup excluded.
    for text in texts[:3]:
        predictor.predict_probabilities(text)
    seconds = []
    for index in range(repeats):
        start = time.perf_counter()
        predictor.predict_probabilities(texts[index % len(texts)])
        seconds.append(time.perf_counter() - start)
    return {"device": "cpu", "batch_size": 1, "warmup_calls": min(3, len(texts)), "measured_calls": repeats,
            "median_ms": float(np.median(seconds) * 1000), "p95_ms": float(np.percentile(seconds, 95) * 1000),
            "includes_preprocessing_and_tokenization": True, "threads": torch.get_num_threads()}


def export_model(kind, data, config, output_root, selection, verification=False):
    from .inference import ComplaintPredictor
    output_root = Path(output_root)
    export = output_root / "deployment_model"
    if export.exists():
        shutil.rmtree(export)
    export.mkdir(parents=True)
    model = load_checkpoint(kind, output_root / "models" / kind, data, config, "cpu").eval()
    for filename in ["labels.json", "preprocessing.json"]:
        shutil.copy2(data.root / filename, export / filename)
    if kind == "distilbert":
        model.save_pretrained(export / "model", safe_serialization=True)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(data.root / "tokenizer", local_files_only=True)
        tokenizer.save_pretrained(export / "model")
    else:
        torch.save(model.state_dict(), export / "model.pt")
        write_json(export / "architecture.json", model.architecture)
        shutil.copy2(data.root / "vocabulary.json", export / "vocabulary.json")
    write_json(export / "selection.json", selection)
    bundle = {"schema_version": 1, "model": kind, "training_status": "trained", "verification_only": verification,
              "preparation_id": data.manifest["preparation_id"], "max_input_characters": 20000,
              "probabilities_are_calibrated": False, "source_text_was_preprocessed_upstream": True,
              "file_sha256": {str(path.relative_to(export)): sha256(path) for path in sorted(export.rglob("*")) if path.is_file()}}
    write_json(export / "bundle.json", bundle)
    predictor = ComplaintPredictor(export, device="cpu", allow_verification=verification)
    indices = data.indices("validation")[:min(8, len(data.indices("validation")))]
    loader = make_loader(data, "validation", kind, config, torch.device("cpu"), indices)
    weights = torch.tensor(data.preprocessing["class_weights"], dtype=torch.float32)
    expected, positions, _ = predict_loader(model, kind, loader, torch.device("cpu"), weights)
    actual = np.stack([predictor.predict_probabilities(data.records.iloc[index].narrative) for index in positions])
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
    measured = benchmark_bundle(predictor, data.records.iloc[indices].narrative.tolist(), config["benchmark_repeats"])
    measured["recurrent_max_length"] = config["recurrent_max_length"]
    measured["transformer_max_length"] = config["transformer_max_length"]
    measured["validation_reload_max_probability_difference"] = float(np.abs(actual - expected).max())
    write_json(output_root / "cpu_inference_benchmark.json", measured)
    del model, predictor
    gc.collect()
    return export


def export_selected(data, config, output_root):
    output_root = Path(output_root)
    selection = read_json(output_root / "selection.json")
    signature, _ = experiment_signature(data, config)
    if selection["signature"] != signature:
        raise RuntimeError("Selection signature mismatch")
    if config["final_test"] and not (output_root / "final_test/complete.json").exists():
        raise RuntimeError("Complete the final test phase before final delivery")
    export = export_model(selection["selected_model"], data, config, output_root, selection)
    # Standalone app package, without source data or optimizer/cache files.
    app_root = output_root / "gradio_app"
    if app_root.exists():
        shutil.rmtree(app_root)
    app_root.mkdir()
    shutil.copytree(export, app_root / "deployment_model")
    source_root = Path(__file__).resolve().parents[2]
    shutil.copy2(source_root / "app.py", app_root / "app.py")
    shutil.copy2(source_root / "requirements.txt", app_root / "requirements.txt")
    package_dir = app_root / "src/complaints"
    package_dir.mkdir(parents=True)
    for name in ["__init__.py", "text.py", "models.py", "inference.py"]:
        shutil.copy2(Path(__file__).parent / name, package_dir / name)
    (app_root / "README.md").write_text(
        "---\ntitle: Consumer Complaint Classifier\nsdk: gradio\nsdk_version: 5.49.1\napp_file: app.py\n---\n\n"
        "# Consumer Complaint Classifier\n\nRun `pip install -r requirements.txt` then `python app.py`. "
        "The selected model and exact inference preprocessing are included. No training at startup.\n",
        encoding="utf-8")
    shutil.make_archive(str(output_root / "consumer_complaints_gradio_app"), "zip", app_root)
    # Retain all four best checkpoints, reports, code and run configuration for reproducibility.
    for destination, subdir in [("training_source", "src")]:
        target = output_root / destination
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source_root / subdir, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    write_json(output_root / "experiment_config.json", config)
    write_json(output_root / "environment.json", environment_record())
    write_json(output_root / "prepared_input_reference.json", data.manifest)
    # Archive paths are explicit: never accidentally include the archive itself or the HF cache.
    archive = output_root.parent / f"{output_root.name}_results.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for path in sorted(output_root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.name != "consumer_complaints_gradio_app.zip":
                if any(part in path.relative_to(output_root).parts for part in ["gradio_app", "deployment_model", "verification_only"]):
                    continue
                z.write(path, str(path.relative_to(output_root)))
        z.write(output_root / "consumer_complaints_gradio_app.zip", "consumer_complaints_gradio_app.zip")
    reports_archive = output_root.parent / f"{output_root.name}_reports.zip"
    with zipfile.ZipFile(reports_archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output_root.rglob("*")):
            relative = path.relative_to(output_root)
            if path.is_file() and path.suffix not in {".pt", ".zip", ".pyc"} and not any(
                part in relative.parts for part in ["gradio_app", "deployment_model", "verification_only", "__pycache__"]
            ):
                z.write(path, str(relative))
    return {"reports_archive": str(reports_archive), "results_archive": str(archive),
            "app_archive": str(output_root / "consumer_complaints_gradio_app.zip")}


def verification_indices(data, split, per_class=2):
    indices = []
    allowed = data.indices(split)
    for label in range(len(data.label_names)):
        indices.extend(allowed[data.targets[allowed] == label][:per_class].tolist())
    return np.array(sorted(indices), dtype=np.int64)


def run_verification(data, config, output_root, device, kinds=None):
    kinds = REQUIRED_MODELS if kinds is None else list(kinds)
    smoke = {**config, "recurrent_epochs": 1, "transformer_epochs": 1, "recurrent_batch_size": 5,
             "transformer_batch_size": 2, "eval_batch_size": 5, "gradient_accumulation_steps": 1,
             "benchmark_repeats": 3, "warmup_ratio": 0.0}
    root = Path(output_root) / "verification_only"
    results = {}
    for kind in kinds:
        results[kind] = run_model(kind, data, smoke, root / kind, torch.device(device), verification=True,
                                  train_indices=verification_indices(data, "train", 1),
                                  validation_indices=verification_indices(data, "validation", 1))
        export_model(kind, data, smoke, root / kind, {"verification_only": True, "model": kind}, verification=True)
    write_json(root / "verification_summary.json", {"status": "passed", "verification_only": True, "models": results,
                                                    "note": "Tiny functional checks, not experimental scores."})
    write_json(Path(output_root) / "functional_checks.json", {"status": "passed", "verification_only": True,
                                                            "models_checked": list(results), "note": "Tiny checks only; verification weights discarded."})
    shutil.rmtree(root)
    return results
