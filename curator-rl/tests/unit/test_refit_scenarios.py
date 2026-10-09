"""The refit scenarios (D-88) are reproducible from the measurements and load cleanly."""

import importlib.util
from pathlib import Path

import pytest

from curator_rl.simulator.scenarios import load_scenario

REPO_ROOT = Path(__file__).resolve().parents[2]


def _refit_module():
    path = REPO_ROOT / "experiments" / "analysis" / "refit_scenarios.py"
    spec = importlib.util.spec_from_file_location("refit_scenarios", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_kappa_fit_and_cost_bracket_match_the_committed_scenarios():
    mod = _refit_module()
    kappa, per_env = mod.fit_kappa()
    assert kappa == pytest.approx(2.14, abs=0.01)
    assert all(1.0 < k < 3.0 for k in per_env.values())
    r1, r2 = mod.unit_costs(0.0), mod.unit_costs(0.5)
    spread = {name: max(c[e] for e in mod.LEARNABLE) / min(c[e] for e in mod.LEARNABLE)
              for name, c in (("r1", r1), ("r2", r2))}
    assert spread["r1"] > spread["r2"] > 2.0          # both ends of the bracket meet the 2x rule
    for name, costs in (("scenario_si_r1.yaml", r1), ("scenario_si_r2.yaml", r2)):
        sc = load_scenario(REPO_ROOT / "configs" / "sim" / name)
        assert sc.world.rollout_concentration == pytest.approx(kappa, abs=0.005)
        for e in sc.envs:
            assert e.cost_usd_per_prompt == pytest.approx(costs[e.env_id], rel=1e-4)


def test_refit_scenarios_match_the_level1_portfolio():
    for name in ("scenario_si_r1.yaml", "scenario_si_r2.yaml", "scenario_sj_r.yaml"):
        sc = load_scenario(REPO_ROOT / "configs" / "sim" / name)
        assert [e.env_id for e in sc.envs] == ["gsm8k", "math35", "mbpp", "countdown", "noisy"]
        weights = {e.env_id: e.bench_weight for e in sc.envs}
        assert weights["noisy"] == 0.0 and sum(weights.values()) == pytest.approx(1.0)
    sj = load_scenario(REPO_ROOT / "configs" / "sim" / "scenario_sj_r.yaml")
    assert len({e.cost_usd_per_prompt for e in sj.envs}) == 1          # homogeneous twin
