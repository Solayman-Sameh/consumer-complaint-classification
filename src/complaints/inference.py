"""CPU-capable inference from an exported bundle, without training data."""
import hashlib
import json
from pathlib import Path

import torch

from .models import RecurrentClassifier
from .text import clean_recurrent, encode_recurrent, normalize_text


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class ComplaintPredictor:
    def __init__(self, bundle_dir, device="cpu", allow_verification=False):
        self.root = Path(bundle_dir)
        self.device = torch.device(device)
        self.bundle = load_json(self.root / "bundle.json")
        if self.bundle["training_status"] != "trained":
            raise ValueError("Bundle is not a trained export")
        if self.bundle.get("verification_only") and not allow_verification:
            raise ValueError("Functional-check weights cannot be used as a final trained application.")
        for relative, expected in self.bundle["file_sha256"].items():
            target = self.root / relative
            if not target.exists():
                if "safetensors" in relative or "model.pt" in relative:
                    raise FileNotFoundError(
                        f"Trained model weights missing at '{target}'. "
                        f"On a fresh clone, run 'python scripts/download_model.py' to fetch and verify them."
                    )
                raise FileNotFoundError(f"Required inference file missing: {relative}")
            with target.open("rb") as f:
                actual = hashlib.file_digest(f, "sha256").hexdigest()
            if actual != expected:
                raise RuntimeError(f"Inference file failed integrity check: {relative}")
        self.kind = self.bundle["model"]
        self.preprocessing = load_json(self.root / "preprocessing.json")
        self.labels = load_json(self.root / "labels.json")
        self.names = [self.labels["id2label"][str(i)] for i in range(len(self.labels["id2label"]))]
        if self.kind == "distilbert":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self.model = AutoModelForSequenceClassification.from_pretrained(self.root / "model", local_files_only=True, attn_implementation="eager")
            self.tokenizer = AutoTokenizer.from_pretrained(self.root / "model", local_files_only=True)
        else:
            architecture = load_json(self.root / "architecture.json")
            self.model = RecurrentClassifier(**architecture)
            self.model.load_state_dict(torch.load(self.root / "model.pt", map_location="cpu", weights_only=True))
            self.vocabulary = load_json(self.root / "vocabulary.json")
            self.stopwords = set(self.preprocessing["stopwords"])
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict_probabilities(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Enter a complaint before classifying.")
        if len(text) > 20000:
            raise ValueError("Please shorten the complaint to 20,000 characters or fewer.")
        if not any(character.isalpha() for character in text):
            raise ValueError("Enter a complaint containing words.")
        if self.kind == "distilbert":
            batch = self.tokenizer(normalize_text(text), truncation=True,
                                   max_length=self.preprocessing["transformer_max_length"], return_tensors="pt")
            batch = {key: value.to(self.device) for key, value in batch.items() if key in {"input_ids", "attention_mask"}}
            logits = self.model(**batch).logits
        else:
            ids = encode_recurrent(text, self.vocabulary, self.stopwords, self.preprocessing["recurrent_max_length"])
            logits = self.model(torch.tensor([ids], dtype=torch.long, device=self.device), torch.tensor([len(ids)], dtype=torch.long))
        return logits.float().softmax(dim=-1)[0].cpu().numpy()

    def input_length(self, text):
        """Untruncated input length and cap, in this model's token units."""
        if self.kind == "distilbert":
            return len(self.tokenizer(normalize_text(text), truncation=False, verbose=False)["input_ids"]), self.preprocessing["transformer_max_length"]
        return len(clean_recurrent(text, self.stopwords).split()), self.preprocessing["recurrent_max_length"]

    def predict(self, text):
        probabilities = self.predict_probabilities(text)
        return {name: float(probabilities[i]) for i, name in enumerate(self.names)}
