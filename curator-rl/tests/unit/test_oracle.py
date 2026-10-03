"""Unit tests for the oracles (Roadmap F.4, v1 Phase 3 item 10)."""

import numpy as np
import pytest

from curator_rl.simulator.oracle import (
    DPOracle,
    MyopicOracle,
    StaticOracle,
    deterministic_final_score,
    simplex_grid,
)
from curator_rl.simulator.world import SimEnvParams, SimWorld


def make_world(envs: list[SimEnvParams], seed: int = 0, **kwargs) -> SimWorld:
    defaults = dict(
        steps_per_round=5, prompts_per_round=80, group_size=8, budget_usd=2.0,
        skill_noise_std=0.0, cost_lognormal_sigma=0.0,
    )
    defaults.update(kwargs)
    return SimWorld(envs, **defaults, rng=np.random.default_rng(seed))


def sample_envs() -> list[SimEnvParams]:
    return [
        SimEnvParams(env_id="a", difficulty=0.2, eta=0.012, cost_usd_per_prompt=0.001),
        SimEnvParams(env_id="b", difficulty=0.8, eta=0.012, cost_usd_per_prompt=0.001),
        SimEnvParams(env_id="junk", eta=0.01, cost_usd_per_prompt=0.001, noisy_q=0.3),
    ]


def test_simplex_grid_sums_to_one():
    grid = simplex_grid(3, 0.25)
    assert len(grid) == 15  # C(n + k - 1, k) with n=3, k=4
    for w in grid:
        assert abs(sum(w) - 1.0) < 1e-12
        assert all(x >= 0 for x in w)


def test_dp_dominates_every_grid_mixture():
    world = make_world(sample_envs(), seed=0)
    dp = DPOracle(world, skill_levels=24, budget_units=30, simplex_step=0.25)
    best = dp.value()
    for action in dp.actions:
        mixture = dict(zip(dp.order, action))
        v_fixed = dp.fixed_policy_value(mixture)
        assert best >= v_fixed - 1e-9, f"DP {best} < fixed {v_fixed} at {mixture}"


def test_dp_beats_uniform_value_substantially():
    world = make_world(sample_envs(), seed=0)
    dp = DPOracle(world, skill_levels=24, budget_units=30, simplex_step=1 / 3)
    v_uniform = dp.fixed_policy_value({e: 1 / 3 for e in dp.order})
    assert dp.value() > v_uniform + 0.05


def test_dp_value_close_to_continuous_dynamics():
    envs = sample_envs()
    world = make_world(envs, seed=0)
    dp = DPOracle(world, skill_levels=48, budget_units=50, simplex_step=0.25)
    mixture = {"a": 0.0, "b": 1.0, "junk": 0.0}
    v_quantised = dp.fixed_policy_value(mixture)
    v_continuous = deterministic_final_score(
        envs, lambda _s, _t: dict(mixture),
        steps_per_round=world.R, prompts_per_round=world.M,
        budget_usd=world.budget_usd, max_rounds=500,
    )
    assert abs(v_quantised - v_continuous) < 0.02


def test_dp_rejects_unsupported_worlds():
    four = [
        SimEnvParams(env_id=str(i), difficulty=0.5, eta=0.01, cost_usd_per_prompt=0.001)
        for i in range(4)
    ]
    with pytest.raises(ValueError, match="N <= 3"):
        DPOracle(make_world(four))
    drifted = [SimEnvParams(env_id="a", difficulty=0.5, eta=0.01,
                            cost_usd_per_prompt=0.001, drift_round=3, drift_shift=0.5)]
    with pytest.raises(ValueError, match="drift"):
        DPOracle(make_world(drifted))
    coupling = [SimEnvParams(env_id="a", difficulty=0.5, eta=0.01,
                             cost_usd_per_prompt=0.001, cost_skill_slope=0.5)]
    with pytest.raises(ValueError, match="cost_skill_slope"):
        DPOracle(make_world(coupling))


def test_static_oracle_beats_uniform():
    envs = sample_envs()
    world = make_world(envs, seed=0)
    oracle = StaticOracle(world, simplex_step=0.25, seed=0)
    v_uniform = deterministic_final_score(
        envs, lambda _s, _t: {e: 1 / 3 for e in oracle.env_ids},
        steps_per_round=world.R, prompts_per_round=world.M,
        budget_usd=world.budget_usd, max_rounds=500,
    )
    assert oracle.best_value > v_uniform
    assert abs(sum(oracle.best_weights.values()) - 1.0) < 1e-9
    assert all(w >= 0 for w in oracle.best_weights.values())
    # the oracle must not touch the live world
    assert world.skills == {e.env_id: e.skill0 for e in envs}


def test_myopic_weights_valid_and_sensible():
    world = make_world(sample_envs(), seed=0)
    oracle = MyopicOracle(world)
    w = oracle.select_mixture(None)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v >= 0 for v in w.values())
    assert w["junk"] == 0.0  # noisy env gets nothing
    assert w["a"] + w["b"] > 0.99


def test_myopic_respects_cost():
    cheap = SimEnvParams(env_id="cheap", difficulty=0.5, eta=0.012,
                         cost_usd_per_prompt=0.0005)
    dear = SimEnvParams(env_id="dear", difficulty=0.5, eta=0.012,
                        cost_usd_per_prompt=0.004)
    world = make_world([cheap, dear])
    w = MyopicOracle(world).select_mixture(None)
    assert w["cheap"] > w["dear"]
