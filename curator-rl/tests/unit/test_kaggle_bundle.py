"""The Kaggle bundle must never contain the sealed test set or raw data (Roadmap D.3, v3 4.9)."""

from __future__ import annotations

import importlib.util
import zipfile

import pytest
from tests.conftest import REPO_ROOT


def _load_module():
    spec = importlib.util.spec_from_file_location("make_kaggle_bundle", REPO_ROOT / "scripts" / "make_kaggle_bundle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_repo(root):
    for rel in (
        "src/curator_rl/__init__.py",
        "src/curator_rl/__pycache__/x.cpython-314.pyc",
        "configs/base.yaml",
        "scripts/kaggle_pilot.py",
        "data/manifests/a.txt",
        "data/processed/gsm8k/dev.jsonl",
        "data/test_sealed/gsm8k.jsonl",
        "data/raw/gsm8k/train.parquet",
        "pyproject.toml",
    ):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")


def test_collect_excludes_sealed_raw_and_caches(tmp_path, monkeypatch):
    mod = _load_module()
    _fake_repo(tmp_path)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    names = {f.relative_to(tmp_path).as_posix() for f in mod.collect()}
    assert "data/processed/gsm8k/dev.jsonl" in names and "src/curator_rl/__init__.py" in names
    assert not any("test_sealed" in n or "data/raw" in n or n.endswith(".pyc") for n in names)


def test_bundle_refuses_to_include_the_sealed_test_set(tmp_path, monkeypatch):
    mod = _load_module()
    _fake_repo(tmp_path)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "INCLUDE_DIRS", (*mod.INCLUDE_DIRS, "data/test_sealed"))
    with pytest.raises(SystemExit, match="forbidden"):
        mod.main(["--out", str(tmp_path / "out.zip")])
    assert not (tmp_path / "out.zip").exists()


def test_real_bundle_has_no_sealed_data(tmp_path):
    mod = _load_module()
    out = tmp_path / "bundle.zip"
    assert mod.main(["--out", str(out)]) == 0
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    assert "BUNDLE_INFO.json" in names
    assert not any("test_sealed" in n or "data/raw" in n for n in names)
