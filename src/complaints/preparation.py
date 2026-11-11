"""Reproduce audited preparation inside Kaggle from a verified CSV."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from complaints.text import clean_recurrent, normalize_text, text_hash, PREPROCESSING_VERSION

SOURCE_SHA256 = "45e406e64bba3a77dc76a66b096eb54c7b5e5a896be4dd851503e505465c6599"
SOURCE_SLUG = "shashwatwork/consume-complaints-dataset-fo-nlp"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "prepared_data")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "configs/experiment.json").read_text())
    (ROOT / "reports/figures").mkdir(parents=True, exist_ok=True)
    source_csv_sha = sha(args.csv)
    if source_csv_sha != "a120ac8450cb3839bf0085bbae68f7c0f3b5351302cafd0bc85ae27a251010b4":
        raise RuntimeError("Source CSV differs from audited version 1; preparation stopped.")
    raw = pd.read_csv(args.csv, keep_default_na=False)
    if raw.columns.tolist() != ["Unnamed: 0", "product", "narrative"]:
        raise RuntimeError("Unexpected source schema")
    frame = raw.rename(columns={"Unnamed: 0": "source_id"}).copy()
    if not frame.source_id.is_unique:
        raise RuntimeError("Source row IDs are not unique")
    frame["narrative"] = frame.narrative.map(normalize_text)
    frame["product"] = frame["product"].astype(str).str.strip()
    frame["text_sha256"] = frame.narrative.map(text_hash)
    stopwords = set(ENGLISH_STOP_WORDS) - {"no", "not", "nor", "never", "none", "cannot", "without"}
    frame["recurrent_text"] = frame.narrative.map(lambda t: clean_recurrent(t, stopwords))
    frame["recurrent_sha256"] = frame.recurrent_text.map(text_hash)
    frame["reason"] = "retained"
    frame["canonical_source_id"] = frame.source_id
    counts = [{"stage": "source", "remaining": len(frame), "excluded": 0}]

    def exclude(mask, reason):
        mask = mask & frame.reason.eq("retained")
        n = int(mask.sum())
        frame.loc[mask, "reason"] = reason
        counts.append({"stage": reason, "remaining": int(frame.reason.eq("retained").sum()), "excluded": n})

    exclude(frame.narrative.eq("") | frame["product"].eq(""), "blank_text_or_label")
    active = frame.loc[frame.reason.eq("retained")]
    conflicts = active.groupby("text_sha256")["product"].nunique()
    conflicting_keys = set(conflicts[conflicts.gt(1)].index)
    exclude(frame.text_sha256.isin(conflicting_keys), "conflicting_source_text_labels")
    active = frame.loc[frame.reason.eq("retained")]
    canonical = active.groupby("text_sha256").source_id.min()
    frame.loc[active.index, "canonical_source_id"] = active.text_sha256.map(canonical)
    exclude(frame.reason.eq("retained") & frame.source_id.ne(frame.canonical_source_id), "duplicate_source_text")
    exclude(frame.recurrent_text.eq(""), "empty_after_recurrent_cleaning")
    active = frame.loc[frame.reason.eq("retained")]
    rec_conflicts = active.groupby("recurrent_sha256")["product"].nunique()
    rec_conflict_keys = set(rec_conflicts[rec_conflicts.gt(1)].index)
    exclude(frame.recurrent_sha256.isin(rec_conflict_keys), "conflicting_recurrent_text_labels")
    active = frame.loc[frame.reason.eq("retained")]
    rec_canonical = active.groupby("recurrent_sha256").source_id.min()
    frame.loc[active.index, "canonical_source_id"] = active.recurrent_sha256.map(rec_canonical)
    exclude(frame.reason.eq("retained") & frame.source_id.ne(frame.canonical_source_id), "duplicate_recurrent_text")
    data = frame.loc[frame.reason.eq("retained")].sort_values("source_id").copy().reset_index(drop=True)
    labels = sorted(data["product"].unique().tolist())
    label2id = {label: i for i, label in enumerate(labels)}
    data["label_id"] = data["product"].map(label2id)
    tokenizer = AutoTokenizer.from_pretrained(cfg["transformer_checkpoint"], revision=cfg["transformer_revision"])
    tokenizer.save_pretrained(out / "tokenizer")
    AutoConfig.from_pretrained(cfg["transformer_checkpoint"], revision=cfg["transformer_revision"]).save_pretrained(out / "tokenizer")
    transformer = np.zeros((len(data), cfg["transformer_max_length"]), dtype=np.uint16)
    hf_lengths = np.zeros(len(data), dtype=np.uint16)
    hf_full_lengths = np.zeros(len(data), dtype=np.int32)
    for start in range(0, len(data), 1024):
        texts = data.narrative.iloc[start:start + 1024].tolist()
        all_ids = tokenizer(texts, truncation=False, padding=False, verbose=False)["input_ids"]
        enc = tokenizer(texts, truncation=True, max_length=cfg["transformer_max_length"], padding="max_length", return_tensors="np")
        end = start + len(texts)
        transformer[start:end] = enc["input_ids"].astype(np.uint16)
        hf_lengths[start:end] = enc["attention_mask"].sum(axis=1).astype(np.uint16)
        hf_full_lengths[start:end] = [len(ids) for ids in all_ids]
        if start % 20480 == 0:
            print(f"Tokenized {end:,}/{len(data):,}", flush=True)
    # Group complaints that become identical after either model's head truncation.
    # These keys use fixed text rules / a fixed pretrained tokenizer, not learned vocabulary.
    parents = np.arange(len(data))
    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return int(x)
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)
    prefix_keys = data.recurrent_text.map(lambda text: text_hash(" ".join(text.split()[:cfg["recurrent_max_length"]])))
    hf_keys = [hashlib.sha256(row.astype("<i4").tobytes()).hexdigest() for row in transformer]
    for keys in [prefix_keys, hf_keys]:
        seen = {}
        for index, key in enumerate(keys):
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index
    # Unknown-token substitution can create further equal inputs. Monotonically
    # merge those groups, re-split and refit the vocabulary from the new training
    # partition until actual encoded inputs are isolated. No vocabulary fitted
    # on a provisional partition is ever shipped or used for training.
    for split_pass in range(10):
        data["split_group_id"] = [int(data.source_id.iloc[find(i)]) for i in range(len(data))]
        folds = np.full(len(data), -1, dtype=np.int8)
        splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=cfg["seed"])
        for fold, (_, indices) in enumerate(splitter.split(np.zeros(len(data)), data.label_id, groups=data.split_group_id)):
            folds[indices] = fold
        data["split"] = np.where(folds == 0, "validation", np.where(folds == 1, "test", "train"))
        train = data.loc[data.split.eq("train")]
        counter = Counter(word for text in train.recurrent_text for word in text.split())
        ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        vocabulary = {"<PAD>": 0, "<UNK>": 1}
        vocabulary.update({word: i + 2 for i, (word, _) in enumerate(ordered[:cfg["vocabulary_cap"] - 2])})
        recurrent = np.zeros((len(data), cfg["recurrent_max_length"]), dtype=np.uint16)
        rec_lengths = np.zeros(len(data), dtype=np.uint16)
        rec_full_lengths = np.zeros(len(data), dtype=np.int32)
        unk_counts = {split: 0 for split in ["train", "validation", "test"]}
        word_counts = {split: 0 for split in unk_counts}
        for i, row in enumerate(data.itertuples()):
            words = row.recurrent_text.split()
            ids = [vocabulary.get(word, 1) for word in words]
            rec_full_lengths[i] = len(ids)
            ids = ids[:cfg["recurrent_max_length"]]
            rec_lengths[i] = len(ids)
            recurrent[i, :len(ids)] = ids
            unk_counts[row.split] += sum(x == 1 for x in ids)
            word_counts[row.split] += len(ids)
        data["recurrent_input_sha256"] = [hashlib.sha256(row.astype("<i4").tobytes()).hexdigest() for row in recurrent]
        cross_split = data.groupby("recurrent_input_sha256").split.nunique().gt(1)
        print(f"Split verification pass {split_pass + 1}: {int(cross_split.sum())} recurrent input collisions across splits", flush=True)
        if not cross_split.any():
            break
        seen = {}
        for index, key in enumerate(data.recurrent_input_sha256):
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index
    else:
        raise RuntimeError("Encoded input grouping did not converge; no prepared split is accepted")
    data["prepared_index"] = np.arange(len(data))
    split_by_id = data.set_index("source_id")["split"]
    frame["split"] = frame.source_id.map(split_by_id).fillna("")
    for column in ["text_sha256", "recurrent_sha256"]:
        assert data[column].is_unique, column
    data["transformer_input_sha256"] = hf_keys
    for column in ["split_group_id", "recurrent_input_sha256", "transformer_input_sha256"]:
        assert data.groupby(column).split.nunique().max() == 1, f"Input overlap across splits: {column}"
    np.savez_compressed(out / "encoded_inputs.npz", recurrent_ids=recurrent, recurrent_lengths=rec_lengths,
                        transformer_ids=transformer, transformer_lengths=hf_lengths,
                        recurrent_full_lengths=rec_full_lengths, transformer_full_lengths=hf_full_lengths)
    # Recompute fingerprints from verified bytes at load time rather than ship
    # 496k redundant SHA strings. This reduces handoff bandwidth substantially.
    keep = ["prepared_index", "source_id", "product", "label_id", "split", "split_group_id", "narrative"]
    data[keep].to_csv(out / "records.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    ledger_columns = ["source_id", "product", "reason", "canonical_source_id", "split"]
    frame[ledger_columns].to_csv(out / "source_row_ledger.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    split_counts = pd.crosstab(data["product"], data.split).reindex(labels)
    split_counts.to_csv(ROOT / "reports/class_counts_by_split.csv")
    class_counts = np.bincount(train.label_id, minlength=len(labels))
    weights = len(train) / (len(labels) * class_counts)
    dump(out / "labels.json", {"label2id": label2id, "id2label": {str(i): name for name, i in label2id.items()}})
    dump(out / "vocabulary.json", vocabulary)
    dump(out / "preprocessing.json", {
        "version": PREPROCESSING_VERSION, "stopwords": sorted(stopwords), "negations_preserved": ["no", "not", "nor", "never", "none", "cannot", "without"],
        "source_text": "Already processed upstream; lowercase alphabetic words only. Raw narratives unavailable.",
        "recurrent_pipeline": "NFKC, casefold, replace non a-z with spaces, fixed stopwords, training-only vocabulary, head truncation",
        "transformer_pipeline": "NFKC/casefold/whitespace source normalization only, pinned pretrained tokenizer, head truncation",
        "class_weights": weights.tolist(), "class_weight_policy": "N / (K * training_class_count)",
        "recurrent_max_length": cfg["recurrent_max_length"], "transformer_max_length": cfg["transformer_max_length"],
    })
    dump(out / "experiment.json", cfg)
    length_summary = {}
    for split in ["train", "validation", "test"]:
        mask = data.split.eq(split).to_numpy()
        length_summary[split] = {
            "records": int(mask.sum()),
            "recurrent_length_percentiles": {str(q): float(np.percentile(rec_full_lengths[mask], q)) for q in [50, 90, 95, 99, 100]},
            "transformer_length_percentiles": {str(q): float(np.percentile(hf_full_lengths[mask], q)) for q in [50, 90, 95, 99, 100]},
            "recurrent_truncation_fraction": float(np.mean(rec_full_lengths[mask] > cfg["recurrent_max_length"])),
            "transformer_truncation_fraction": float(np.mean(hf_full_lengths[mask] > cfg["transformer_max_length"])),
            "recurrent_unknown_token_fraction": unk_counts[split] / max(word_counts[split], 1),
        }
    source_counts = frame["product"].value_counts().to_dict()
    audit = {
        "source": SOURCE_SLUG, "source_version": 1, "retrieved_date": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).date().isoformat(), "historical_source_archive_sha256": SOURCE_SHA256,
        "source_csv_sha256": source_csv_sha, "source_rows": len(raw), "source_columns": raw.columns.tolist(),
        "source_class_counts": source_counts, "official_splits": "None: one CSV with no split column",
        "normalized_conflicting_groups": len(conflicting_keys), "recurrent_conflicting_groups_after_source_dedup": len(rec_conflict_keys),
        "exclusion_flow": counts, "retained_rows": len(data), "labels": labels,
        "split_counts": {k: int(v) for k, v in data.split.value_counts().items()},
        "class_counts_by_split": split_counts.to_dict(), "lengths": length_summary,
        "vocabulary_size_including_specials": len(vocabulary), "training_distinct_words": len(counter),
        "training_class_weights": {label: float(weights[i]) for i, label in enumerate(labels)},
        "duplicate_overlap_between_splits": {"normalized_source_text": 0, "recurrent_cleaned_text": 0, "recurrent_encoded_input": 0, "transformer_encoded_input": 0},
        "split_policy": "10 StratifiedGroupKFold folds with seed 42: fold 0 validation, fold 1 test, other folds train. Truncated-input equivalence groups stay together.",
        "split_group_count": int(data.split_group_id.nunique()),
        "multirow_split_groups": int(data.groupby("split_group_id").size().gt(1).sum()),
        "encoded_collision_grouping_passes": split_pass + 1,
        "source_preprocessing_evidence": {
            "uppercase_narratives": int(raw.narrative.str.contains(r"[A-Z]", regex=True).sum()),
            "digit_narratives": int(raw.narrative.str.contains(r"\d", regex=True).sum()),
            "non_lowercase_letter_or_whitespace_narratives": int(raw.narrative.str.contains(r"[^a-z\s]", regex=True).sum()),
            "raw_narratives_available": False,
        },
        "near_duplicate_policy": "Exact normalized and recurrent-cleaned duplicates removed. Semantic/paraphrase near-duplicates are not guaranteed absent.",
        "upstream_limitations": "Unknown upstream cleaning/label consolidation; no raw text or original complaint identifiers. No original-data linkage is invented.",
        "training_status": "NOT TRAINED",
        "preparation_versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "transformers", "tokenizers"]},
    }
    dump(out / "audit.json", audit)
    dump(ROOT / "reports/dataset_audit.json", audit)
    train.groupby("product", sort=True).head(2)[["source_id", "product", "narrative"]].to_csv(ROOT / "reports/training_examples.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    split_counts[["train", "validation", "test"]].plot.bar(stacked=True, ax=axes[0], color=["#2563eb", "#14b8a6", "#f59e0b"])
    axes[0].set_title("Retained complaints by category and split")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=25, labelsize=8)
    axes[0].set_ylabel("Complaints")
    train_mask = data.split.eq("train").to_numpy()
    axes[1].hist(hf_full_lengths[train_mask], bins=np.arange(0, 801, 20), color="#2563eb", alpha=.8)
    axes[1].axvline(256, color="#dc2626", linestyle="--", label="256-token cap")
    axes[1].set_title("Training narrative lengths (DistilBERT tokens)")
    axes[1].set_xlabel("Tokens including special tokens; view capped at 800")
    axes[1].set_ylabel("Complaints")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures/data_overview.png", dpi=170)
    plt.close(fig)
    files = {str(p.relative_to(out)): {"sha256": sha(p), "size_bytes": p.stat().st_size}
             for p in sorted(out.rglob("*")) if p.is_file() and p.name != "prepared_manifest.json"}
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {"schema_version": 1, "preparation_id": identity, "historical_source_archive_sha256": SOURCE_SHA256,
                "record_count": len(data), "preprocessing_version": PREPROCESSING_VERSION, "files": files}
    dump(out / "prepared_manifest.json", manifest)
    print(json.dumps({"retained": len(data), "splits": audit["split_counts"], "preparation_id": identity,
                      "lengths": length_summary, "exclusion_flow": counts}, indent=2), flush=True)


if __name__ == "__main__":
    main()
