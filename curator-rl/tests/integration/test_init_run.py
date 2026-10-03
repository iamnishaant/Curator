"""Integration: init-run end to end (Roadmap Phase 1 integration test)."""

from pathlib import Path

import yaml

from conftest import BASE_CFG
from curator_rl.cli import init_run
from curator_rl.core.config import CuratorConfigError


def _config_in(tmp_path: Path, overrides: dict | None = None) -> Path:
    payload = yaml.safe_load(BASE_CFG.read_text(encoding="utf-8"))
    payload["experiment"]["name"] = "initint"
    payload["paths"] = {"runs_root": str(tmp_path / "runs")}
    for dotted, value in (overrides or {}).items():
        node = payload
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    path = tmp_path / f"cfg_{abs(hash(str(overrides)))}.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_init_run_creates_frozen_run_directory(tmp_path):
    cfg_path = _config_in(tmp_path)
    result = init_run(cfg_path)

    assert set(result) >= {"run_id", "run_dir", "config_hash"}
    run_dir = Path(result["run_dir"])
    assert run_dir.exists()
    for sub in ("logs", "checkpoints", "reports"):
        assert (run_dir / sub).exists()
    assert (run_dir / "config.yaml").exists()

    meta_text = (run_dir / "metadata.json").read_text(encoding="utf-8")
    assert '"config_hash"' in meta_text and "git" in meta_text


def test_init_run_idempotent_for_same_config(tmp_path):
    cfg_path = _config_in(tmp_path)
    first = init_run(cfg_path)
    second = init_run(cfg_path)
    assert second["run_id"] == first["run_id"]
    assert second["config_hash"] == first["config_hash"]
    assert second["reused"] is True


def test_init_run_returns_error_on_conflicting_config(tmp_path, monkeypatch):
    """Same run id, different config => hard error (Roadmap: refuses a conflicting config)."""
    cfg_path = _config_in(tmp_path)
    result = init_run(cfg_path)
    run_dir = Path(result["run_dir"])

    # tamper with the frozen config so its hash differs while the run id stays identical
    frozen = run_dir / "config.yaml"
    payload = yaml.safe_load(frozen.read_text(encoding="utf-8"))
    payload["scheduler"]["tau"] = 0.7
    frozen.write_text(yaml.safe_dump(payload), encoding="utf-8")

    try:
        init_run(cfg_path)
    except CuratorConfigError as exc:
        assert "conflict" in str(exc)
    else:
        raise AssertionError("a conflicting config for an existing run id must be refused")
