"""Unit tests for the SignalEngine facade (Roadmap Phase 4)."""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.conftest import BASE_CFG, make_round_obs

from curator_rl.signals import SignalEngine


def make_engine(base_cfg, envs=("a", "b", "c")) -> SignalEngine:
    return SignalEngine(
        base_cfg.signals, base_cfg.proxy, base_cfg.calib,
        base_cfg.scheduler.cost_exponent, base_cfg.group_size, envs,
    )


def test_engine_construction_kwargs(base_cfg):
    with pytest.raises(ValueError):
        make_engine(base_cfg, envs=())
    with pytest.raises(ValueError):
        make_engine(base_cfg, envs=("a", "a"))


def test_unknown_env_id_rejected(base_cfg):
    engine = make_engine(base_cfg)
    bad = make_round_obs(("a", "b", "zzz"), round_t=1)
    with pytest.raises(ValueError):
        engine.update(bad)


def test_first_round_observation_stream(base_cfg):
    engine = make_engine(base_cfg)
    vectors = engine.update(make_round_obs(("a", "b", "c"), round_t=1))
    assert set(vectors) == {"a", "b", "c"}
    for sv in vectors.values():
        assert sv.status.value == "S1_unexplored"
        assert sv.pass_rate > 0.0
        assert sv.n_rounds_seen == 1
        assert sv.unit_cost_usd > 0.0  # interim cost source (D-33)


def test_status_progression_easy_env_reaches_s3(base_cfg):
    """An env with a high constant success rate must reach S3 after n_min."""
    # 26 prompts/round, lambda=0.9: n_groups_eff = 26 * (1-0.9^t)/0.1
    engine = make_engine(base_cfg)
    last = None
    for t in range(1, 40):
        vectors = engine.update(make_round_obs(("a", "b", "c"), round_t=t,
                                               prompts_per_env=26, success_rate=0.9))
        last = vectors["a"].status
    assert last.value == "S3_saturated"


def test_plateau_env_stays_learning(base_cfg):
    """Flat mid pass rates: never S3 (no saturation), never S4 (richness high)."""
    engine = make_engine(base_cfg)
    trace = []
    for t in range(1, 40):
        vectors = engine.update(make_round_obs(("a", "b", "c"), round_t=t,
                                               prompts_per_env=26, success_rate=0.5))
        trace.append(vectors["a"].status)
    assert trace[-1].value == "S2_learning"
    assert all(s.name in ("S1", "S2") for s in trace)


def test_too_hard_env_reaches_s4(base_cfg):
    """Near-zero pass rate AND low richness -> S4 (mix: few groups carry successes)."""
    engine = make_engine(base_cfg)
    from tests.conftest import make_env_obs

    from curator_rl.core.types import RoundObservation
    trace = []
    for t in range(1, 30):
        obs_env = make_env_obs("a", n_prompts=26, group_size=8, success_rate=0.01)
        obs_env = obs_env.__class__(  # low richness: 3 of 26 prompts got k>0
            env_id="a", n_prompts=26, n_rollouts=26 * 8, k_success=2, n_groups_mixed=3,
            sum_score=2 / 8.0, sum_score_sq=2 * (1 / 8.0) ** 2,
            prompt_tokens=26 * 256, completion_tokens=26 * 8 * 128,
            verifier_seconds=1e-4 * 26 * 8, gpu_seconds=0.001 * 3600, cost_usd=0.001,
        )
        obs = RoundObservation(
            round=t, steps=t * 5,
            per_env={"a": obs_env, "b": make_env_obs("b", n_prompts=26, success_rate=0.5),
                     "c": make_env_obs("c", n_prompts=26, success_rate=0.5)},
            weights_used={"a": 1 / 3, "b": 1 / 3, "c": 1 / 3},
            round_cost_usd=0.004, overhead_usd=0.0, budget_remaining_usd=9.0,
        )
        trace.append(engine.update(obs)["a"].status)
    assert trace[-1].value == "S4_too_hard"


# permutation equivariance (hypothesis, first use in the repo) ------------------

@settings(max_examples=25, deadline=None)
@given(permutation=st.permutations(["a", "b", "c"]))
def test_engine_is_permutation_equivariant(permutation):

    from curator_rl.core.config import RootConfig, load_config  # noqa: F401
    from curator_rl.core.types import RoundObservation

    cfg = load_config(BASE_CFG)
    engine_a = SignalEngine(cfg.signals, cfg.proxy, cfg.calib, cfg.scheduler.cost_exponent,
                            cfg.group_size, ("a", "b", "c"))
    engine_b = SignalEngine(cfg.signals, cfg.proxy, cfg.calib, cfg.scheduler.cost_exponent,
                            cfg.group_size, ("a", "b", "c"))
    obs1 = make_round_obs(("c", "a", "b"), round_t=3, prompts_per_env=12)
    per_env = dict(obs1.per_env)
    obs2 = RoundObservation(
        round=3, steps=15, per_env={e: per_env[e] for e in permutation},
        weights_used=obs1.weights_used, round_cost_usd=obs1.round_cost_usd,
        overhead_usd=0.0, budget_remaining_usd=9.0,
    )
    vec_a = engine_a.update(obs1)
    vec_b = engine_b.update(obs2)
    for env in ("a", "b", "c"):
        assert vec_a[env] == vec_b[env]


def test_checkpoint_round_trip_reproduces_status(base_cfg):
    engine = make_engine(base_cfg)
    for t in range(1, 8):
        engine.update(make_round_obs(("a", "b", "c"), round_t=t, prompts_per_env=26, success_rate=0.9))
    state = engine.get_state()
    forked = make_engine(base_cfg)
    forked.load_checkpoint(json.loads(json.dumps(state)))  # JSON-safety (Part M item 13)
    next_obs = make_round_obs(("a", "b", "c"), round_t=8, prompts_per_env=26, success_rate=0.9)
    v1 = engine.update(next_obs)["a"]
    v2 = forked.update(next_obs)["a"]
    assert (v1.status, v1.status_note, v1.lp, v1.proxy_reward) == (v2.status, v2.status_note, v2.lp, v2.proxy_reward)


def test_statuses_accessor(base_cfg):
    engine = make_engine(base_cfg)
    engine.update(make_round_obs(("a", "b", "c"), round_t=1))
    assert engine.statuses("a").value == "S1_unexplored"
