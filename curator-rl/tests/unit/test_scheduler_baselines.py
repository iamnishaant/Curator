"""Unit tests for BaseScheduler and the Uniform/Random stubs (Phase 3)."""

import numpy as np
import pytest

from curator_rl.core.seeding import SeedManager
from curator_rl.scheduler.base import validate_weights
from curator_rl.scheduler.baselines.random_baseline import RandomScheduler
from curator_rl.scheduler.baselines.uniform import UniformScheduler

ENVS = ("a", "b", "c")


def test_uniform_is_one_over_n():
    scheduler = UniformScheduler(ENVS)
    w = scheduler.select_mixture(None)
    assert w == {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    validate_weights(w, ENVS)  # must not raise


def test_random_mixture_is_valid_and_deterministic():
    s1 = RandomScheduler(ENVS, SeedManager(7).rng("scheduler"))
    s2 = RandomScheduler(ENVS, SeedManager(7).rng("scheduler"))
    w1 = s1.select_mixture(None)
    w2 = s2.select_mixture(None)
    assert w1 == w2  # same seed stream -> same draw
    validate_weights(w1, ENVS)
    # redraws differ
    assert s1.select_mixture(None) != s1.select_mixture(None)


def test_random_mixture_covers_simplex():
    scheduler = RandomScheduler(ENVS, np.random.default_rng(0))
    draws = [scheduler.select_mixture(None) for _ in range(200)]
    mean = {e: sum(d[e] for d in draws) / len(draws) for e in ENVS}
    for e in ENVS:
        assert abs(mean[e] - 1 / 3) < 0.05  # Dirichlet(1) is exchangeable


def test_validate_weights_rejects_bad_mixtures():
    with pytest.raises(ValueError):
        validate_weights({"a": 0.5, "b": 0.5}, ENVS)  # missing env
    with pytest.raises(ValueError):
        validate_weights({"a": 0.5, "b": 0.6, "c": -0.1}, ENVS)  # negative
    with pytest.raises(ValueError):
        validate_weights({"a": 0.5, "b": 0.4, "c": 0.2}, ENVS)  # sum != 1


def test_scheduler_needs_envs():
    with pytest.raises(ValueError):
        UniformScheduler(())


def _scenario(sizes, bench=None):
    from curator_rl.simulator.scenarios import ScenarioCfg, SimEnvCfg

    envs = [
        SimEnvCfg(
            env_id=f"e{i}", eta=0.01, cost_usd_per_prompt=0.001,
            nominal_size=None if sizes is None else sizes[i],
            bench_weight=None if bench is None else bench[i],
        )
        for i in range(3)
    ]
    return ScenarioCfg(scenario_id="T", description="t", budget_usd=1.0, envs=envs)


def test_static_weights_are_size_proportional_when_sizes_declared():
    from experiments.run_sim import _static_weights  # noqa: PLC0415

    w = _static_weights(_scenario([6000, 3000, 1000]))
    assert w == pytest.approx({"e0": 0.6, "e1": 0.3, "e2": 0.1})


def test_static_weights_fall_back_to_bench_weights_then_uniform():
    from experiments.run_sim import _static_weights  # noqa: PLC0415

    by_bench = _static_weights(_scenario(None, bench=[0.5, 0.25, 0.25]))
    assert by_bench == pytest.approx({"e0": 0.5, "e1": 0.25, "e2": 0.25})
    uniform = _static_weights(_scenario(None))
    assert uniform == pytest.approx({"e0": 1 / 3, "e1": 1 / 3, "e2": 1 / 3})


def test_gate2_scenarios_declare_non_degenerate_static_mixtures():
    # D-63: Static must differ from Uniform on the Gate 2 scenarios
    from experiments.run_sim import _static_weights  # noqa: PLC0415
    from tests.conftest import REPO_ROOT  # noqa: PLC0415

    from curator_rl.simulator.scenarios import load_scenario  # noqa: PLC0415

    for sid in ("sa", "sb", "sc"):
        sc = load_scenario(str(REPO_ROOT / "configs" / "sim" / f"scenario_{sid}.yaml"))
        w = _static_weights(sc)
        assert max(w.values()) - min(w.values()) > 0.05
