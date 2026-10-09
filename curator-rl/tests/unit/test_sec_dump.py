"""Unit tests for the SEC-style and DUMP-style baselines and the concentration diagnostic (D-67, D-68)."""

from __future__ import annotations

import math

import pytest
from tests.conftest import BASE_CFG, make_env_obs

from curator_rl.core.config import load_config
from curator_rl.core.types import RoundObservation
from curator_rl.scheduler.base import validate_weights
from curator_rl.scheduler.baselines.dump import DUMPStyleUCB
from curator_rl.scheduler.baselines.sec import SECStyleBandit
from curator_rl.simulator.harness import weight_concentration

ENVS = ("a", "b", "c")


def _cfg():
    return load_config(BASE_CFG)


def _obs(round_t, adv_by_env, prompts=10):
    """RoundObservation where env e has mean |A| = adv_by_env[e] per prompt."""
    from dataclasses import replace

    per_env = {}
    for e, adv in adv_by_env.items():
        base = make_env_obs(e, n_prompts=prompts)
        per_env[e] = replace(base, sum_abs_adv=adv * prompts)
    return RoundObservation(
        round=round_t, steps=5 * round_t, per_env=per_env,
        weights_used={e: 1 / len(per_env) for e in per_env},
        round_cost_usd=0.1, overhead_usd=0.0, budget_remaining_usd=1.0,
    )


# SEC -------------------------------------------------------------------------

def test_sec_td0_update_hand_computed():
    cfg = _cfg()
    sched = SECStyleBandit(ENVS, cfg.baselines.sec.model_copy(update={"alpha": 0.5}), cfg.scheduler)
    sched.update_observation(_obs(1, {"a": 0.8, "b": 0.4, "c": 0.0}))
    assert sched.get_state()["q"] == pytest.approx({"a": 0.4, "b": 0.2, "c": 0.0})
    sched.update_observation(_obs(2, {"a": 0.8, "b": 0.4, "c": 0.0}))
    # Q <- 0.5*0.8 + 0.5*0.4 = 0.6
    assert sched.get_state()["q"]["a"] == pytest.approx(0.6)


def test_sec_only_updates_arms_that_were_pulled():
    cfg = _cfg()
    sched = SECStyleBandit(ENVS, cfg.baselines.sec, cfg.scheduler)
    obs = _obs(1, {"a": 0.8, "b": 0.4, "c": 0.9})
    from dataclasses import replace

    obs = replace(obs, per_env={**obs.per_env, "c": replace(obs.per_env["c"], n_prompts=0, sum_abs_adv=0.0)})
    sched.update_observation(obs)
    assert sched.get_state()["q"]["c"] == 0.0


def test_sec_warmup_is_uniform_then_prefers_high_advantage_arm():
    cfg = _cfg()
    sched = SECStyleBandit(ENVS, cfg.baselines.sec.model_copy(update={"alpha": 1.0, "tau": 0.2}), cfg.scheduler)
    w = None
    for t in range(1, cfg.scheduler.warmup_rounds + 1):
        w = sched.select_mixture(None if t == 1 else _obs(t - 1, {"a": 0.9, "b": 0.1, "c": 0.1}))
        assert w == pytest.approx({e: 1 / 3 for e in ENVS})
        sched.update_observation(_obs(t, {"a": 0.9, "b": 0.1, "c": 0.1}))
    w = sched.select_mixture(_obs(7, {"a": 0.9, "b": 0.1, "c": 0.1}))
    validate_weights(w, ENVS)
    assert w["a"] > w["b"] and w["a"] > 0.6
    floor = cfg.scheduler.epsilon / 3
    assert min(w.values()) >= floor - 1e-12


def test_sec_checkpoint_roundtrip_gives_identical_next_decision():
    cfg = _cfg()
    a = SECStyleBandit(ENVS, cfg.baselines.sec, cfg.scheduler)
    for t in range(1, 10):
        a.select_mixture(_obs(t, {"a": 0.9, "b": 0.3, "c": 0.5}) if t > 1 else None)
        a.update_observation(_obs(t, {"a": 0.9, "b": 0.3, "c": 0.5}))
    b = SECStyleBandit(ENVS, cfg.baselines.sec, cfg.scheduler)
    b.load_checkpoint(a.get_state())
    obs = _obs(10, {"a": 0.2, "b": 0.8, "c": 0.5})
    assert a.select_mixture(obs) == b.select_mixture(obs)


# DUMP ------------------------------------------------------------------------

def test_dump_scores_hand_computed():
    cfg = _cfg()
    sched = DUMPStyleUCB(ENVS, cfg.baselines.dump.model_copy(update={"exploration_coef": 2.0}), cfg.scheduler)
    sched.update_observation(_obs(1, {"a": 0.8, "b": 0.4, "c": 0.0}, prompts=10))
    sched.update_observation(_obs(2, {"a": 0.6, "b": 0.4, "c": 0.0}, prompts=10))
    s = sched.scores()
    log_arg = math.log(60.0)        # N = 3 arms * 20 prompts
    assert s["a"] == pytest.approx(0.7 + 2.0 * math.sqrt(log_arg / 20.0))
    assert s["c"] == pytest.approx(0.0 + 2.0 * math.sqrt(log_arg / 20.0))


def test_dump_bonus_shrinks_as_an_arm_is_pulled_more():
    cfg = _cfg()
    sched = DUMPStyleUCB(ENVS, cfg.baselines.dump, cfg.scheduler)
    sched.update_observation(_obs(1, {"a": 0.5, "b": 0.5, "c": 0.5}, prompts=4))
    before = sched.scores()["a"]
    for t in range(2, 8):
        sched.update_observation(_obs(t, {"a": 0.5, "b": 0.5, "c": 0.5}, prompts=40))
    assert sched.scores()["a"] < before


def test_dump_weights_valid_after_warmup_and_checkpoint_roundtrip():
    cfg = _cfg()
    a = DUMPStyleUCB(ENVS, cfg.baselines.dump, cfg.scheduler)
    for t in range(1, 12):
        w = a.select_mixture(_obs(t, {"a": 0.9, "b": 0.3, "c": 0.0}) if t > 1 else None)
        validate_weights(w, ENVS)
        a.update_observation(_obs(t, {"a": 0.9, "b": 0.3, "c": 0.0}))
    b = DUMPStyleUCB(ENVS, cfg.baselines.dump, cfg.scheduler)
    b.load_checkpoint(a.get_state())
    obs = _obs(12, {"a": 0.9, "b": 0.3, "c": 0.0})
    assert a.select_mixture(obs) == b.select_mixture(obs)


# Concentration ---------------------------------------------------------------

def test_weight_concentration_hand_computed():
    logs = [
        {"weights": {"a": 1.0, "b": 0.0}},                  # one-hot: max 1, entropy 0
        {"weights": {"a": 0.5, "b": 0.5}},                  # uniform: max 0.5, entropy 1, not > 0.5
    ]
    c = weight_concentration(logs)
    assert c["mean_max_weight"] == pytest.approx(0.75)
    assert c["norm_entropy"] == pytest.approx(0.5)
    assert c["frac_rounds_over_half"] == pytest.approx(0.5)
    assert weight_concentration(logs, skip_rounds=1)["mean_max_weight"] == pytest.approx(0.5)
    assert weight_concentration([])["n_rounds"] == 0
