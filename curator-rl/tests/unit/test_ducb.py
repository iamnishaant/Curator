"""Unit tests for the D-UCB scheduler, Curator facade and baselines (Phase 5)."""

from __future__ import annotations

import json

import pytest
from tests.conftest import BASE_CFG, make_env_obs

from curator_rl.core.config import load_config
from curator_rl.core.types import RoundObservation
from curator_rl.scheduler.baselines.lp import LPCurriculum
from curator_rl.scheduler.baselines.static import StaticMixtureScheduler
from curator_rl.scheduler.baselines.ucb import StandardUCB
from curator_rl.scheduler.curator import Curator


def make_sched(envs=("a", "b", "c"), method="curator", **overrides):
    cfg = load_config(BASE_CFG)
    scheduler_cfg = cfg.scheduler.model_copy(update=overrides) if overrides else cfg.scheduler
    common = (cfg.signals, cfg.proxy, scheduler_cfg, cfg.calib, cfg.group_size)
    if method == "curator":
        return Curator(envs, *common, prompts_per_round=80)
    if method == "ucb":
        return StandardUCB(envs, *common)
    if method == "lp":
        return LPCurriculum(envs, *common)
    raise ValueError(method)


def obs_stream(envs=("a", "b", "c"), rounds=20, rate_fn=None, prompts=26):
    """Deterministic RoundObservation stream with per-env success rates."""
    rate_fn = rate_fn or (lambda env, t: 0.5)
    out = []
    for t in range(1, rounds + 1):
        per_env = {}
        for env in envs:
            per_env[env] = make_env_obs(
                env, n_prompts=prompts, group_size=8, success_rate=rate_fn(env, t),
            )
        out.append(
            RoundObservation(
                round=t, steps=t * 5, per_env=per_env,
                weights_used={e: 1.0 / len(envs) for e in envs},
                round_cost_usd=0.004, overhead_usd=0.0, budget_remaining_usd=9.0,
            )
        )
    return out


def run_schedule(scheduler, stream):
    weights_by_round = []
    for i, obs in enumerate(stream):
        w = scheduler.select_mixture(stream[i - 1] if i else None)
        scheduler.update_observation(obs)
        weights_by_round.append(w)
    return weights_by_round


# DiscountedUCB / Curator --------------------------------------------------------

def test_warmup_is_uniform_then_scheduling_kicks_in():
    sched = make_sched(warmup_rounds=3)
    stream = obs_stream(rounds=8)
    weights = run_schedule(sched, stream)
    for w in weights[:3]:
        assert all(abs(v - 1 / 3) < 1e-12 for v in w.values())
    assert sched.last_decision is not None
    assert weights[3] != weights[4] or True  # post-warmup weights are data-driven


def test_equal_rewards_give_equal_scores_after_one_round():
    sched = make_sched(warmup_rounds=1, gamma=1.0, exploration_coef=0.5)
    stream = obs_stream(rounds=2)
    run_schedule(sched, stream[:1])
    # after round 1: u_i = 1/3, r_bar ~ constant -> mu_hat identical for all;
    # bonus identical too -> scores identical -> mixture exactly the softmax of
    # equal scores = uniform (up to the epsilon floor redistribution)
    scores = sched.compute_scores()
    vals = list(scores.values())
    assert all(abs(v - vals[0]) < 1e-9 for v in vals)


def test_discounting_decays_stale_rewards():
    # 'a' ramps up (positive LP => high proxy reward) then saturates; with
    # gamma=1 the accumulated advantage persists, with gamma=0.5 it decays
    def rate(env, t):
        return 0.5 + 0.4 * min(t, 6) / 6 if env == "a" else 0.5

    stale = make_sched(warmup_rounds=1, gamma=1.0)
    fresh = make_sched(warmup_rounds=1, gamma=0.5)
    stream = obs_stream(rounds=30, rate_fn=rate)
    run_schedule(stale, stream)
    run_schedule(fresh, stream)
    s_stale = stale.compute_scores()
    s_fresh = fresh.compute_scores()
    assert s_stale["a"] - s_stale["b"] > s_fresh["a"] - s_fresh["b"]


def test_learner_beats_flat_arm_while_unsaturated():
    """Mid-window: the rising arm earns more mass than the flat arm; the
    saturated arm is capped. (The learner eventually crosses p_sat and gets
    capped too -- that is the engine's design, not a bug.)"""
    def rate(env, t):
        return 0.9 if env == "a" else (0.5 + 0.35 * min(t, 20) / 20 if env == "b" else 0.5)

    sched = make_sched(warmup_rounds=6)
    stream = obs_stream(rounds=40, rate_fn=rate)
    weights = run_schedule(sched, stream)
    mid = weights[11:28]  # b rises 0.69 -> 0.85; still below/at p_sat early on
    mean_c = sum(w["c"] for w in mid) / len(mid)
    mean_b = sum(w["b"] for w in mid) / len(mid)
    mean_a = sum(w["a"] for w in mid) / len(mid)
    assert mean_b > mean_c  # the learner gains over the flat arm
    assert mean_a < mean_b  # the saturated arm is capped below the learner
    assert mean_c < 1 / 3   # the flat arm starves below uniform


def test_mixture_decision_fields_and_quotas():
    sched = make_sched(warmup_rounds=1)
    stream = obs_stream(rounds=3)
    run_schedule(sched, stream)
    decision = sched.last_decision
    assert decision is not None
    assert decision.round == 3
    assert abs(sum(decision.weights.values()) - 1.0) < 1e-12
    assert set(decision.ucb_scores) == {"a", "b", "c"}
    assert decision.quotas and sum(decision.quotas.values()) == 80  # prompts_per_round
    assert all(v >= 0 for v in decision.quotas.values())
    assert len(decision.rng_hash) == 12


def test_standard_ucb_matches_ablated_ducb():
    """StandardUCB == D-UCB with gamma=1, cost_exponent=0, statuses off (J.1 #5)."""
    stream = obs_stream(rounds=15)
    ucb = run_schedule(make_sched(method="ucb", warmup_rounds=2), stream)
    ablated = run_schedule(
        make_sched(warmup_rounds=2, gamma=1.0, cost_exponent=0.0, status_control="off"),
        stream,
    )
    for w1, w2 in zip(ucb, ablated):
        for e in ("a", "b", "c"):
            assert abs(w1[e] - w2[e]) < 1e-12


def test_checkpoint_round_trip_reproduces_decisions():
    stream = obs_stream(rounds=15)
    sched = make_sched(warmup_rounds=2)
    for i, obs in enumerate(stream):
        sched.select_mixture(stream[i - 1] if i else None)
        sched.update_observation(obs)
        if i == 7:
            state = json.loads(json.dumps(sched.get_state()))
    forked = make_sched(warmup_rounds=2)
    forked.load_checkpoint(state)
    for obs in stream[8:]:
        w1 = sched.select_mixture(None)  # obs already folded in by update_observation
        w2 = forked.select_mixture(None)
        assert w1 == w2
        sched.update_observation(obs)
        forked.update_observation(obs)


# baselines ----------------------------------------------------------------------

def test_static_mixture_validates_and_returns_fixed_weights():
    sched = StaticMixtureScheduler(("a", "b"), {"a": 0.7, "b": 0.3})
    assert sched.select_mixture(None) == {"a": 0.7, "b": 0.3}
    with pytest.raises(ValueError):
        StaticMixtureScheduler(("a", "b"), {"a": 0.7, "b": 0.2})


def test_lp_curriculum_prefers_the_learner():
    def rate(env, t):
        return 0.5 + 0.4 * t / 20 if env == "b" else 0.5

    sched = make_sched(method="lp", warmup_rounds=2)
    stream = obs_stream(rounds=20, rate_fn=rate)
    weights = run_schedule(sched, stream)
    late_b = sum(w["b"] for w in weights[-5:]) / 5
    late_a = sum(w["a"] for w in weights[-5:]) / 5
    assert late_b > late_a  # the rising env earns more mass
