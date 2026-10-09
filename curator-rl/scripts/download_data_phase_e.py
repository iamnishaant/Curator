"""Download MATH and MBPP raw files (Roadmap v3 Phase E, D.2).

MATH (training pool and sealed test source):
  DigitalLearningGmbH/MATH-lighteval  data/train, data/test   (7500 / 5000 problems with level and solution)
  HuggingFaceH4/MATH-500              test.jsonl              (the sealed test set for the math domain)
MBPP:
  google-research-datasets/mbpp       full/{train,validation,test,prompt}.parquet

Files go to data/raw/math/ and data/raw/mbpp/ with SHA-256 and revision in metadata.json.
Idempotent: an existing file is kept. Usage: python scripts/download_data_phase_e.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCES = {
    "math": {
        "dataset": "DigitalLearningGmbH/MATH-lighteval",
        "files": {"train.parquet": "data/train-00000-of-00001.parquet",
                  "test.parquet": "data/test-00000-of-00001.parquet"},
    },
    "math500": {
        "dataset": "HuggingFaceH4/MATH-500",
        "files": {"test.jsonl": "test.jsonl"},
        "target": "math",
    },
    "mbpp": {
        "dataset": "google-research-datasets/mbpp",
        "files": {f"{s}.parquet": f"full/{s}-00000-of-00001.parquet"
                  for s in ("train", "validation", "test", "prompt")},
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def revision(dataset: str) -> str:
    try:
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{dataset}", timeout=30) as r:
            return str(json.loads(r.read().decode("utf-8")).get("sha", "main"))
    except Exception:  # noqa: BLE001 - offline fallback keeps the run possible
        return "main"


def main() -> int:
    meta: dict[str, dict] = {}
    for key, spec in SOURCES.items():
        target = REPO_ROOT / "data" / "raw" / spec.get("target", key)
        target.mkdir(parents=True, exist_ok=True)
        rec: dict[str, str] = {"dataset_id": spec["dataset"], "revision": revision(spec["dataset"])}
        for name, rel in spec["files"].items():
            dest = target / (f"math500_{name}" if key == "math500" else name)
            if not dest.exists():
                url = f"https://huggingface.co/datasets/{spec['dataset']}/resolve/main/{rel}"
                print(f"downloading {url}")
                with urllib.request.urlopen(url, timeout=120) as resp, dest.open("wb") as out:
                    out.write(resp.read())
                time.sleep(0.5)
            rec[f"{dest.name}_sha256"] = sha256(dest)
        rec["retrieved_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta[key] = rec
    for folder in ("math", "mbpp"):
        records = {k: v for k, v in meta.items() if SOURCES[k].get("target", k) == folder}
        (REPO_ROOT / "data" / "raw" / folder / "metadata.json").write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
