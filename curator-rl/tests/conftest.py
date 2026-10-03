from pathlib import Path

import pytest
import yaml

from curator_rl.core import config as config_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CFG = REPO_ROOT / "configs" / "base.yaml"
SMOKE_CFG = REPO_ROOT / "configs" / "experiment" / "smoke.yaml"


@pytest.fixture
def base_cfg() -> config_mod.RootConfig:
    return config_mod.load_config(BASE_CFG)


@pytest.fixture
def base_payload() -> dict:
    return yaml.safe_load(BASE_CFG.read_text(encoding="utf-8"))


def make_config_file(tmp_path: Path, overrides: dict | None = None, name: str = "cfg.yaml") -> Path:
    payload = yaml.safe_load(BASE_CFG.read_text(encoding="utf-8"))
    if overrides:
        for dotted, value in overrides.items():
            node = payload
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def assert_conflict_raises(config_path: Path, overrides: list[str], exc_cls: type = ValueError):
    with pytest.raises(exc_cls):
        config_mod.load_config(config_path, overrides)


__all__ = ["REPO_ROOT", "BASE_CFG", "SMOKE_CFG", "base_cfg", "base_payload", "make_config_file", "config_mod"]
