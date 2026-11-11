"""Download and verify the deployment model weights for fresh repository clones.

GitHub rejects files larger than 100 MB, so the 268 MB DistilBERT safetensors
weights are excluded from Git. This utility downloads the verified model weights
from the hosted Hugging Face release and verifies their SHA-256 against bundle.json.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = ROOT / "deployment_model/bundle.json"
MODEL_DIR = ROOT / "deployment_model/model"
TARGET_FILE = MODEL_DIR / "model.safetensors"

DEFAULT_SOURCE_URL = os.environ.get(
    "MODEL_SOURCE_URL",
    "https://huggingface.co/Solayman-Sameh/consumer-complaint-classifier/resolve/main/model.safetensors",
)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def download_with_progress(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(".tmp")
    print(f"Downloading model weights from:\n  {url}")
    print(f"Destination: {destination}")

    try:
        with urllib.request.urlopen(url) as response, temp_path.open("wb") as out_file:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1 MB chunks
            while chunk := response.read(chunk_size):
                out_file.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    percent = (downloaded / total) * 100
                    mb = downloaded / (1024 * 1024)
                    total_mb = total / (1024 * 1024)
                    sys.stdout.write(f"\rProgress: {mb:.1f} MB / {total_mb:.1f} MB ({percent:.1f}%)")
                    sys.stdout.flush()
                else:
                    mb = downloaded / (1024 * 1024)
                    sys.stdout.write(f"\rDownloaded: {mb:.1f} MB")
                    sys.stdout.flush()
        print()
        temp_path.replace(destination)
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Download failed: {exc}") from exc


def main():
    if not BUNDLE_PATH.exists():
        print(f"Error: bundle manifest not found at {BUNDLE_PATH}", file=sys.stderr)
        sys.exit(1)

    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    expected_sha256 = bundle["file_sha256"].get("model/model.safetensors")

    if TARGET_FILE.exists():
        print(f"Model weights file already exists at {TARGET_FILE}")
        print("Verifying SHA-256 integrity against bundle.json...")
        actual_sha256 = compute_sha256(TARGET_FILE)
        if actual_sha256 == expected_sha256:
            print("Verification PASSED: SHA-256 matches bundle.json.")
            print(f"  Hash: {actual_sha256}")
            return
        else:
            print("Checksum mismatch! The existing file is incomplete or corrupted.", file=sys.stderr)
            print(f"  Expected: {expected_sha256}", file=sys.stderr)
            print(f"  Actual:   {actual_sha256}", file=sys.stderr)
            choice = input("Would you like to redownload the weights? [y/N]: ").strip().lower()
            if choice != "y":
                sys.exit(1)

    download_with_progress(DEFAULT_SOURCE_URL, TARGET_FILE)
    print("Verifying downloaded file SHA-256...")
    actual_sha256 = compute_sha256(TARGET_FILE)
    if actual_sha256 == expected_sha256:
        print("Verification PASSED: Downloaded weights match exact manifest checksum.")
        print("The application is now ready to run: python app.py")
    else:
        TARGET_FILE.unlink(missing_ok=True)
        print("Verification FAILED: Checksum does not match bundle.json. File deleted.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
