"""Download the GSM8K parquet files (Roadmap Phase 2, D.2).

Fetches the official `openai/gsm8k` (config `main`) train and test parquet
files directly from the HuggingFace hub into `data/raw/gsm8k/`, records
SHA-256 of each file plus the dataset revision in `metadata.json`.

Usage: python scripts/download_data.py
Idempotent: existing files with matching SHA-256 are not re-downloaded.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_ID = "openai/gsm8k"
FILES = {
    "train": "main/train-00000-of-00001.parquet",
    "test": "main/test-00000-of-00001.parquet",
}
_BASE = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_revision() -> str:
    """Best-effort dataset revision id; 'main' if the API is unreachable."""
    url = f"https://huggingface.co/api/datasets/{DATASET_ID}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return str(payload.get("sha", "main"))
    except Exception:  # noqa: BLE001 - offline fallback keeps the run possible
        return "main"


def download(target_dir: Path, force: bool = False) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, str] = {}
    for split, relpath in FILES.items():
        dest = target_dir / f"{split}.parquet"
        if dest.exists() and not force:
            records[f"{split}_sha256"] = file_sha256(dest)
            continue
        url = f"{_BASE}/{relpath}"
        print(f"downloading {url} -> {dest}")
        with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        records[f"{split}_sha256"] = file_sha256(dest)
        records[f"{split}_url"] = url
        time.sleep(0.5)
    records["dataset_id"] = DATASET_ID
    records["revision"] = fetch_revision()
    records["retrieved_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (target_dir / "metadata.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(records, indent=2))


def main() -> int:
    try:
        download(REPO_ROOT / "data" / "raw" / "gsm8k")
    except Exception as exc:  # noqa: BLE001 - script surface
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
