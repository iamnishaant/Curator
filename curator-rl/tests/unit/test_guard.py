"""Unit tests for the sealed-test guard (Roadmap D.3, Phase 2 skeleton)."""

import json
from pathlib import Path

import pytest

from curator_rl.core.config import load_config
from curator_rl.evaluation.guard import SealedTest, SealedTestError

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cfg():
    return load_config(REPO_ROOT / "configs" / "base.yaml")


@pytest.fixture
def repo(tmp_path, cfg):
    """A fake repo root: reports/frozen_config.hash + a sealed data file."""
    reports = tmp_path / "reports"
    reports.mkdir()
    from curator_rl.core.config import config_hash

    (reports / "frozen_config.hash").write_text(config_hash(cfg) + "\n", encoding="utf-8")
    sealed = tmp_path / "data" / "test_sealed"
    sealed.mkdir(parents=True)
    rows = [
        {"problem_id": "gsm8k-test-0", "question": "q0", "answer": "42"},
        {"problem_id": "gsm8k-test-1", "question": "q1", "answer": "43"},
    ]
    (sealed / "gsm8k.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return tmp_path


def _open(cfg, repo, **kw):
    return SealedTest.open(
        split="test",
        checkpoint_hash="abc123",
        cfg=cfg,
        repo_root=repo,
        **kw,
    )


def test_refused_without_env_var(cfg, repo, monkeypatch):
    monkeypatch.delenv("CURATOR_FINAL_EVAL", raising=False)
    with pytest.raises(SealedTestError, match="CURATOR_FINAL_EVAL"):
        _open(cfg, repo)


def test_refused_for_wrong_split(cfg, repo, monkeypatch):
    monkeypatch.setenv("CURATOR_FINAL_EVAL", "1")
    with pytest.raises(SealedTestError, match="only serves split 'test'"):
        SealedTest.open("calib", "abc123", cfg, repo)


def test_refused_without_frozen_hash(cfg, repo, monkeypatch):
    monkeypatch.setenv("CURATOR_FINAL_EVAL", "1")
    (repo / "reports" / "frozen_config.hash").unlink()
    with pytest.raises(SealedTestError, match="frozen config hash file not found"):
        _open(cfg, repo)


def test_refused_on_config_hash_mismatch(cfg, repo, monkeypatch):
    monkeypatch.setenv("CURATOR_FINAL_EVAL", "1")
    (repo / "reports" / "frozen_config.hash").write_text("deadbeef", encoding="utf-8")
    with pytest.raises(SealedTestError, match="does not match the frozen hash"):
        _open(cfg, repo)


def test_open_writes_access_log_and_loads_items(cfg, repo, monkeypatch):
    monkeypatch.setenv("CURATOR_FINAL_EVAL", "1")
    with _open(cfg, repo) as guard:
        items = guard.load_items("gsm8k")
        assert len(items) == 2
        assert items[0]["problem_id"] == "gsm8k-test-0"
    log = repo / "reports" / "test_access_log.jsonl"
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 1
    assert entries[0]["checkpoint_hash"] == "abc123"
    assert entries[0]["split"] == "test"


def test_second_open_for_same_checkpoint_refused(cfg, repo, monkeypatch):
    monkeypatch.setenv("CURATOR_FINAL_EVAL", "1")
    _open(cfg, repo).close()
    with pytest.raises(SealedTestError, match="second open is refused"):
        _open(cfg, repo)


def test_open_with_different_checkpoint_allowed(cfg, repo, monkeypatch):
    monkeypatch.setenv("CURATOR_FINAL_EVAL", "1")
    _open(cfg, repo).close()
    guard = SealedTest.open(
        split="test", checkpoint_hash="different456", cfg=cfg, repo_root=repo
    )
    guard.close()
    entries = [
        json.loads(line)
        for line in (repo / "reports" / "test_access_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {e["checkpoint_hash"] for e in entries} == {"abc123", "different456"}


def test_missing_sealed_data_file(cfg, repo, monkeypatch):
    monkeypatch.setenv("CURATOR_FINAL_EVAL", "1")
    with _open(cfg, repo) as guard:
        with pytest.raises(FileNotFoundError):
            guard.load_items("countdown")
