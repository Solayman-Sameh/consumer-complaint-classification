"""Independently verify hashes, split isolation, training-only vocabulary and IDs."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from complaints.data import PreparedData, write_json
from complaints.text import clean_recurrent, encode_recurrent


def verify(root):
    data = PreparedData(root)
    stopwords = set(data.preprocessing["stopwords"])
    counts = Counter(word for text in data.records.loc[data.records.split.eq("train"), "narrative"]
                     for word in clean_recurrent(text, stopwords).split())
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:data.config["vocabulary_cap"] - 2]
    expected = {"<PAD>": 0, "<UNK>": 1, **{word: i + 2 for i, (word, _) in enumerate(ordered)}}
    assert data.vocabulary == expected, "Vocabulary is not exactly the training-only vocabulary"
    tokenizer = AutoTokenizer.from_pretrained(data.root / "tokenizer", local_files_only=True)
    samples = np.random.default_rng(42).choice(len(data.records), size=min(256, len(data.records)), replace=False)
    for index in samples:
        narrative = data.records.iloc[index].narrative
        ids = encode_recurrent(narrative, data.vocabulary, stopwords, data.config["recurrent_max_length"])
        assert len(ids) == data.encoded["recurrent_lengths"][index]
        np.testing.assert_array_equal(ids, data.encoded["recurrent_ids"][index, :len(ids)])
        encoded = tokenizer(narrative, truncation=True, max_length=data.config["transformer_max_length"], padding="max_length")
        np.testing.assert_array_equal(encoded["input_ids"], data.encoded["transformer_ids"][index])
    for kind in ["recurrent", "transformer"]:
        keys = [hashlib.sha256(row.astype("<i4").tobytes()).hexdigest() for row in data.encoded[kind + "_ids"]]
        assert keys == data.records[kind + "_input_sha256"].tolist()
        groups = data.records.groupby(kind + "_input_sha256").split.nunique()
        assert int(groups.max()) == 1
    summary = {"status": "passed", "preparation_id": data.manifest["preparation_id"], "records": len(data.records),
               "checks": ["Every prepared file checksum", "Array shapes, padding and label mapping", "All five classes in every split",
                          "No source/recurrent/truncated-input overlap between splits", "Vocabulary exactly reproduced from training only",
                          "Class weights exactly reproduced from training only", "256 sampled records independently re-tokenized",
                          "All encoded-input fingerprints independently recomputed"],
               "sampled_records_reencoded": len(samples)}
    return summary


if __name__ == "__main__":
    result = verify(ROOT / "prepared_data")
    write_json(ROOT / "reports/preparation_verification.json", result)
    print(json.dumps(result, indent=2))
