"""Verify the real CPU bundle and running Gradio app, without training/downloads."""
from pathlib import Path
import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
import sys
import time

os.environ.update(USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="1", GRADIO_ANALYTICS_ENABLED="False")
for variable in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(variable, None)

import numpy as np
import torch
from gradio_client import Client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from app import EXAMPLES
from complaints.inference import ComplaintPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-only", action="store_true", help="Use the tracked reload fixture, as on a fresh clone.")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/application_verification.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    predictor = ComplaintPredictor(ROOT / "deployment_model", device="cpu")
    expected = ["credit_reporting", "debt_collection", "mortgages_and_loans", "credit_card", "retail_banking"]
    natural = []
    for (text,), category in zip(EXAMPLES, expected):
        scores = predictor.predict(text)
        assert len(scores) == 5 and np.isclose(sum(scores.values()), 1, atol=1e-6)
        assert all(np.isfinite(v) and 0 <= v <= 1 for v in scores.values())
        winner = max(scores, key=scores.get)
        assert winner == category, f"Demo prediction mismatch: {text!r}: expected {category}, got {winner}"
        natural.append({"input": text, "expected_category": category, "predicted_category": winner,
                        "probability": scores[winner], "probabilities": scores,
                        "matches_expected": winner == category})

    invalid = []
    for name, value in [("blank", ""), ("whitespace", " \n\t "), ("punctuation", "!!!??..."),
                        ("numbers_only", "123456"), ("over_limit", "a" * 20001), ("non_string", None)]:
        try:
            predictor.predict(value)
            raise AssertionError(f"{name} unexpectedly accepted")
        except ValueError as exc:
            invalid.append({"case": name, "rejected": True, "message": str(exc)})

    long_text = "My bank account has been frozen and I cannot withdraw my money. " * 60
    assert len(long_text) < 20000
    assert len(predictor.predict(long_text)) == 5
    assert len(predictor.predict("I enjoyed the football match and the sunshine today.")) == 5

    # Compare a fresh CPU reload with historical validation probabilities.
    archive_path = ROOT / "reports/models/distilbert/validation/probabilities.npz"
    fixture_path = ROOT / "reports/validation_reload_fixture.json"
    max_difference = 0.0
    reload_count = 0
    if archive_path.exists() and not args.fixture_only:
        archive = np.load(archive_path)
        with (ROOT / "reports/models/distilbert/validation/error_and_correct_examples.csv").open(encoding="utf-8") as f:
            examples = list(csv.DictReader(f))[:5]
        reload_count = len(examples)
        for row in examples:
            index = int(row["prepared_index"])
            position = int(np.flatnonzero(archive["indices"] == index)[0])
            actual = predictor.predict_probabilities(row["narrative"])
            difference = float(np.max(np.abs(actual - archive["probabilities"][position])))
            max_difference = max(max_difference, difference)
            np.testing.assert_allclose(actual, archive["probabilities"][position], atol=2e-5, rtol=2e-4)
    elif fixture_path.exists():
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        reload_count = len(fixture)
        for item in fixture:
            actual = predictor.predict_probabilities(item["narrative"])
            expected = np.array(item["expected_probabilities"], dtype=np.float32)
            difference = float(np.max(np.abs(actual - expected)))
            max_difference = max(max_difference, difference)
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-4)

    else:
        raise FileNotFoundError("No validation reload evidence found. Restore reports/validation_reload_fixture.json.")
    assert reload_count > 0, "Validation reload evidence must contain examples"

    base_url = os.environ.get("APP_TEST_URL", "http://127.0.0.1:7860")
    try:
        client = Client(base_url, verbose=False)
    except Exception as exc:
        raise RuntimeError(f"Cannot reach the Gradio app at {base_url}. Start python app.py in another terminal; set APP_TEST_URL for a different port.") from exc
    api_checks = []
    for row in natural:
        heading, output, status = client.predict(row["input"], api_name="/classify")
        assert "Prediction complete" in status and output.get("confidences")
        assert output["label"] == {"credit_reporting": "Credit reporting", "debt_collection": "Debt collection", "mortgages_and_loans": "Mortgages and loans", "credit_card": "Credit card", "retail_banking": "Retail banking"}[row["expected_category"]]
        api_checks.append({"case": row["expected_category"], "status": "passed"})
    for text in ["", " \n\t ", "!!!", "a" * 20001]:
        heading, output, status = client.predict(text, api_name="/classify")
        assert "Enter a valid complaint" in heading
        assert not output or not output.get("confidences"), output
        api_checks.append({"case": "invalid_input_clears_prediction", "status": "passed"})
    heading, output, status = client.predict(long_text, api_name="/classify")
    assert "256 tokens" in status and output.get("confidences")
    api_checks.append({"case": "long_input_truncation_notice", "status": "passed"})
    heading, output, status = client.predict("I enjoyed the football match and the sunshine today.", api_name="/classify")
    assert output.get("confidences")
    api_checks.append({"case": "unrelated_text_forced_choice_documented", "status": "passed"})

    durations = []
    for i in range(10):
        start = time.perf_counter()
        predictor.predict(EXAMPLES[i % 5][0])
        durations.append((time.perf_counter() - start) * 1000)
    result = {
        "status": "passed", "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": {"python": sys.version.split()[0], "platform": sys.platform,
            "isolated_venv": sys.prefix != sys.base_prefix,
            "host_site_packages_on_sys_path": any("miniconda3" in p.lower() and "site-packages" in p.lower() for p in sys.path),
            "torch_reuse": "Only torch/functorch/torchgen loaded from existing local installation through a restricted importer; other dependencies installed in fresh venv.",
            "versions": {name: metadata.version(name) for name in ["torch", "transformers", "gradio", "huggingface-hub", "tokenizers", "numpy"]}},
        "source_hash_policy": "UTF-8 text with normalized LF newlines",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_text(encoding="utf-8").encode("utf-8")).hexdigest() for name in ["app.py", "src/complaints/inference.py", "src/complaints/text.py", "src/complaints/models.py"]},
        "offline_model_loading": True, "model_weights_downloaded": False,
        "natural_examples": natural, "natural_expected_matches": sum(x["matches_expected"] for x in natural),
        "invalid_input_checks": invalid, "gradio_api_checks": api_checks,
        "validation_reload_examples": reload_count, "reload_source": "fixture" if args.fixture_only or not archive_path.exists() else "archive", "max_probability_difference": max_difference,
        "cpu_median_ms": float(np.median(durations)), "cpu_p95_ms": float(np.percentile(durations, 95)),
        "timing_note": "Ten calls, batch one, two CPU threads, including preprocessing; local hardware only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["status", "natural_expected_matches", "max_probability_difference", "cpu_median_ms"]}))


if __name__ == "__main__":
    main()
