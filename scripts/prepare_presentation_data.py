"""Recover small teaching assets from the already-downloaded prepared archive.

No network, training, or new split assignment. The actual record, vocabulary,
and encoded-array bytes must match the completed experiment's reference.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    reference = json.loads((ROOT / "reports/prepared_input_reference.json").read_text())
    target = ROOT / "reports/presentation"
    target.mkdir(parents=True, exist_ok=True)
    verified = {}
    with zipfile.ZipFile(args.archive) as outer:
        member = next(n for n in outer.namelist() if n.endswith("consumer_complaints_prepared_data.zip"))
        with zipfile.ZipFile(io.BytesIO(outer.read(member))) as prepared:
            def read(name):
                b = prepared.read(next(n for n in prepared.namelist() if n.endswith("/" + name)))
                actual = hashlib.sha256(b).hexdigest()
                assert actual == reference["files"][name]["sha256"], name
                verified[name] = actual
                return b
            records = pd.read_csv(io.BytesIO(read("records.csv.gz")), compression="gzip", keep_default_na=False)
            vocabulary = read("vocabulary.json")
            encoded = np.load(io.BytesIO(read("encoded_inputs.npz")), allow_pickle=False)
            (target / "training_vocabulary.json").write_bytes(vocabulary)
            train = records.loc[records.split.eq("train")].copy()
            train.groupby("product", sort=True).head(2).to_csv(target / "training_examples.csv", index=False)
            mask = records.split.eq("train").to_numpy()
            bins = np.array(list(range(0, 801, 20)) + [1000, 1500, 2000, 2500, 3500, 5000])
            rows = []
            for kind in ["recurrent", "transformer"]:
                hist, edges = np.histogram(encoded[kind + "_full_lengths"][mask], bins=bins)
                rows.extend({"representation": kind, "left": int(l), "right": int(r), "count": int(v)}
                            for l, r, v in zip(edges[:-1], edges[1:], hist))
            pd.DataFrame(rows).to_csv(target / "training_length_histogram.csv", index=False)
    evidence = {"status": "passed", "method": "SHA-256 equality with the completed run's prepared_input_reference.json",
                "completed_preparation_id": reference["preparation_id"], "verified_files": verified,
                "training_records": len(train), "examples": 10,
                "note": "Older container-level preparation metadata differs; all reused experiment data bytes match exactly."}
    (target / "data_recovery_verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
