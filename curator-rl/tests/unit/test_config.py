"""Configuration system tests (Roadmap: Exact tests, items 1-7)."""


import pytest
import yaml

from conftest import BASE_CFG, REPO_ROOT, make_config_file
from curator_rl.core import config as config_mod
from curator_rl.core.config import CuratorConfigError


def test_config_loads_base_yaml(base_cfg):
    cfg = base_cfg
    assert cfg.scheduler.gamma == pytest.approx(0.95)
    assert cfg.signals.lam == pytest.approx(0.9)
    assert cfg.proxy.alpha == pytest.approx(0.5)
    assert cfg.calib.interval_rounds == 5
    assert cfg.steps_per_round == 5 and cfg.prompts_per_step == 16 and cfg.group_size == 8
    assert cfg.experiment.seed == 0


def test_smoke_config_inherits_base(tmp_path):
    from conftest import SMOKE_CFG

    cfg = config_mod.load_config(SMOKE_CFG)
    assert cfg.experiment.name == "smoke"
    assert cfg.scheduler.gamma == pytest.approx(0.95)  # inherited from base.yaml
    assert cfg.budget.total_usd == pytest.approx(0.5)  # overridden


def test_missing_scientific_param_raises(tmp_path):
    base = yaml.safe_load(BASE_CFG.read_text(encoding="utf-8"))
    del base["scheduler"]["gamma"]
    path = tmp_path / "missing_gamma.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    with pytest.raises(CuratorConfigError):
        config_mod.load_config(path)


@pytest.mark.parametrize(
    "dotted, value",
    [
        ("scheduler.gamma", 0.0),          # gamma must be > 0
        ("scheduler.gamma", 1.5),          # gamma must be <= 1
        ("scheduler.epsilon", 1.0),        # epsilon must be < 1
        ("calib.interval_rounds", 0),      # K must be >= 1
        ("scheduler.tau", -1.0),           # tau must be > 0
        ("budget.total_usd", 0.0),         # budget must be > 0
        ("scheduler.score_norm", "magic"), # enum value
        ("signals.lp_method", "lp_z"),     # enum value
    ],
)
def test_invalid_values_raise(tmp_path, dotted, value):
    cfg_path = make_config_file(tmp_path, {dotted: value})
    with pytest.raises(CuratorConfigError):
        config_mod.load_config(cfg_path)


def test_unknown_key_raises(tmp_path):
    cfg_path = make_config_file(tmp_path, {"scheduler": {"typos_key": 1}})
    with pytest.raises(CuratorConfigError):
        config_mod.load_config(cfg_path)


def test_override_syntax(tmp_path):
    cfg_path = make_config_file(tmp_path)
    cfg = config_mod.load_config(cfg_path, ["scheduler.tau=0.5", "experiment.seed=3", "proxy.clip_l=2"])
    assert cfg.scheduler.tau == pytest.approx(0.5)
    assert cfg.experiment.seed == 3
    assert cfg.proxy.clip_l == pytest.approx(2.0)
    with pytest.raises(CuratorConfigError):
        config_mod.load_config(cfg_path, ["does.not.exist=1"])
    with pytest.raises(CuratorConfigError):
        config_mod.load_config(cfg_path, ["scheduler.gamma"])


def test_config_hash_stable_and_sensitive(base_cfg, tmp_path):
    h1 = config_mod.config_hash(base_cfg)
    h2 = config_mod.config_hash(config_mod.load_config(BASE_CFG))
    assert h1 == h2 and len(h1) == 64
    cfg_b = make_config_file(tmp_path, {"scheduler.gamma": 0.90})
    h3 = config_mod.config_hash(config_mod.load_config(cfg_b))
    assert h3 != h1


def test_dump_and_reload_roundtrip(base_cfg, tmp_path):
    out = tmp_path / "dumped.yaml"
    config_mod.dump_config(base_cfg, out)
    reloaded = config_mod.load_config(out)
    assert config_mod.config_hash(reloaded) == config_mod.config_hash(base_cfg)


def test_spec_lint_keys_exist(base_cfg):
    """Every hyperparameter name in the SPEC notation list appears in base.yaml."""
    spec = (REPO_ROOT / "docs" / "SPEC.md").read_text(encoding="utf-8")
    begin = spec.index("BEGIN_HYPERPARAM_LIST")
    end = spec.index("END_HYPERPARAM_LIST")
    block = spec[spec.index("\n", begin) : end]
    keys = [
        line.strip()
        for line in block.splitlines()
        if line.strip() and not line.strip().startswith(("<", "#")) and "." in line
    ]
    assert keys, "SPEC hyperparameter list must not be empty"

    payload = yaml.safe_load(BASE_CFG.read_text(encoding="utf-8"))
    missing = []
    for dotted in keys:
        node = payload
        ok = True
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                ok = False
                break
        if not ok:
            missing.append(dotted)
    assert not missing, f"keys documented in SPEC.md but absent from base.yaml: {missing}"
