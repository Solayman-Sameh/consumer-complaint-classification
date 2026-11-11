"""Strict loading of fully prepared, fingerprinted inputs."""
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .text import clean_recurrent, normalize_text, text_hash


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("Archive member escapes destination")
        z.extractall(destination)


def find_prepared(root="/kaggle/input", extraction_root="/kaggle/working/prepared_input"):
    root = Path(root)
    manifests = sorted(root.rglob("prepared_manifest.json")) if root.exists() else []
    if len(manifests) == 1:
        return manifests[0].parent
    if len(manifests) > 1:
        raise RuntimeError("Multiple prepared datasets found. Set DATA_DIR explicitly in the configuration cell.")
    candidates = sorted(root.rglob("consumer_complaints_prepared_data.zip")) if root.exists() else []
    if len(candidates) != 1:
        raise FileNotFoundError("Attach the supplied consumer_complaints_prepared_data.zip as a Kaggle dataset/input, then rerun. Set DATA_DIR for a custom location.")
    safe_extract(candidates[0], extraction_root)
    manifests = list(Path(extraction_root).rglob("prepared_manifest.json"))
    if len(manifests) != 1:
        raise RuntimeError("Prepared archive must contain exactly one prepared_manifest.json")
    return manifests[0].parent


class EncodedDataset(Dataset):
    def __init__(self, ids, lengths, labels, indices):
        self.ids, self.lengths, self.labels = ids, lengths, labels
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        pos = int(self.indices[index])
        length = int(self.lengths[pos])
        return self.ids[pos, :length].astype(np.int64), length, int(self.labels[pos]), pos


def collate_encoded(rows):
    lengths = torch.tensor([row[1] for row in rows], dtype=torch.long)
    ids = torch.zeros((len(rows), int(lengths.max())), dtype=torch.long)
    for i, row in enumerate(rows):
        ids[i, :row[1]] = torch.from_numpy(row[0])
    return {"input_ids": ids, "lengths": lengths,
            "labels": torch.tensor([row[2] for row in rows], dtype=torch.long),
            "indices": torch.tensor([row[3] for row in rows], dtype=torch.long)}


class PreparedData:
    def __init__(self, root, verify=True):
        self.root = Path(root)
        self.manifest = read_json(self.root / "prepared_manifest.json")
        if self.manifest["schema_version"] != 1:
            raise ValueError("Unsupported prepared data version")
        if verify:
            for relative, expected in self.manifest["files"].items():
                path = self.root / relative
                if path.stat().st_size != expected["size_bytes"] or sha256(path) != expected["sha256"]:
                    raise RuntimeError(f"Prepared input integrity failed: {relative}")
        self.records = pd.read_csv(self.root / "records.csv.gz", keep_default_na=False)
        self.encoded = dict(np.load(self.root / "encoded_inputs.npz", allow_pickle=False))
        self.config = read_json(self.root / "experiment.json")
        self.preprocessing = read_json(self.root / "preprocessing.json")
        self.labels = read_json(self.root / "labels.json")
        self.label_names = [self.labels["id2label"][str(i)] for i in range(len(self.labels["id2label"]))]
        self.vocabulary = read_json(self.root / "vocabulary.json")
        self.targets = self.records.label_id.to_numpy(dtype=np.int64)
        self._validate()

    def _validate(self):
        records = self.records
        n = self.manifest["record_count"]
        assert len(records) == n
        assert np.array_equal(records.prepared_index.to_numpy(), np.arange(n))
        assert records.source_id.is_unique
        records["text_sha256"] = records.narrative.map(lambda text: text_hash(normalize_text(text)))
        stopwords = set(self.preprocessing["stopwords"])
        records["recurrent_sha256"] = records.narrative.map(lambda text: text_hash(clean_recurrent(text, stopwords)))
        for kind in ["recurrent", "transformer"]:
            records[kind + "_input_sha256"] = [hashlib.sha256(row.astype("<i4").tobytes()).hexdigest() for row in self.encoded[kind + "_ids"]]
        assert records.text_sha256.is_unique and records.recurrent_sha256.is_unique
        for column in ["split_group_id", "recurrent_input_sha256", "transformer_input_sha256"]:
            assert records.groupby(column).split.nunique().max() == 1
        assert set(records.split) == {"train", "validation", "test"}
        assert records.narrative.str.strip().ne("").all()
        assert set(self.targets) == set(range(len(self.label_names)))
        assert records["product"].map(self.labels["label2id"]).eq(records.label_id).all()
        for split in ["train", "validation", "test"]:
            assert set(self.targets[self.indices(split)]) == set(range(len(self.label_names)))
        for kind, cap in [("recurrent", self.preprocessing["recurrent_max_length"]),
                          ("transformer", self.preprocessing["transformer_max_length"])]:
            ids = self.encoded[f"{kind}_ids"]
            lengths = self.encoded[f"{kind}_lengths"]
            assert ids.shape == (n, cap)
            assert lengths.shape == (n,)
            assert np.all((lengths > 0) & (lengths <= cap))
            assert np.all(ids >= 0)
            for start in range(0, n, 4096):
                block = ids[start:start + 4096]
                mask = np.arange(cap)[None, :] >= lengths[start:start + 4096, None]
                assert np.all(block[mask] == 0)
        assert int(self.encoded["recurrent_ids"].max()) < len(self.vocabulary)
        counts = np.bincount(self.targets[self.indices("train")], minlength=len(self.label_names))
        expected_weights = counts.sum() / (len(counts) * counts)
        assert np.allclose(expected_weights, self.preprocessing["class_weights"])

    def indices(self, split):
        if split not in {"train", "validation", "test"}:
            raise ValueError(split)
        return np.flatnonzero(self.records.split.to_numpy() == split)

    def dataset(self, split, kind, indices=None):
        prefix = "transformer" if kind == "distilbert" else "recurrent"
        positions = self.indices(split) if indices is None else indices
        if not np.isin(positions, self.indices(split)).all():
            raise ValueError("Dataset indices cross split boundaries")
        return EncodedDataset(self.encoded[f"{prefix}_ids"], self.encoded[f"{prefix}_lengths"], self.targets, positions)
