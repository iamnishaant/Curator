"""Unit tests for the discounted pooled pass-rate tracker (Roadmap E.2, M item)."""

from __future__ import annotations

import math

from tests.conftest import make_env_obs

from curator_rl.signals.passrate import PassRateTracker


def make_tracker(lam=0.5, fast=1.0, slow=0.9, a0=1.0, b0=1.0) -> PassRateTracker:
    return PassRateTracker(lam, fast, slow, a0, b0)


def test_first_round_exact_posterior():
    tracker = make_tracker()
    # k=16 successes of n=32 rollouts over m=4 prompts -> rate 0.5
    post = tracker.update(make_env_obs("e", n_prompts=4, group_size=8, success_rate=0.5))
    assert post.pass_rate == (16.0 + 1.0) / (32.0 + 2.0)
    strength = 32.0 + 2.0
    var = post.pass_rate * (1.0 - post.pass_rate) / (strength + 1.0)
    assert post.pass_lo == post.pass_rate - math.sqrt(var)
    assert post.pass_hi == post.pass_rate + math.sqrt(var)
    assert post.pass_lo < post.pass_rate < post.pass_hi
    assert post.n_groups_eff == 4.0
    assert post.n_rounds_seen == 1
    # group SE: sqrt(p (1-p) / n_groups_eff)
    assert post.se_group == math.sqrt(post.pass_rate * 0.5 / 4.0)


def test_discounting_is_exact_over_rounds():
    tracker = make_tracker(lam=0.5, fast=0.5, slow=0.5)  # all three: easy hand math
    tracker.update(make_env_obs("e", n_prompts=4, group_size=8, success_rate=0.5))  # k=16,n=32
    post = tracker.update(make_env_obs("e", n_prompts=4, group_size=8, success_rate=0.25))  # k=8,n=32
    # S = 0.5*16 + 8 = 16, N = 0.5*32 + 32 = 48 -> p = 17/50
    assert abs(post.pass_rate - 17.0 / 50.0) < 1e-12
    assert post.n_rounds_seen == 2


def test_zero_lambda_window_is_the_latest_round():
    tracker = make_tracker(lam=0.5, fast=0.0, slow=0.5)
    tracker.update(make_env_obs("e", n_prompts=4, group_size=8, success_rate=0.5))
    post = tracker.update(make_env_obs("e", n_prompts=4, group_size=8, success_rate=0.25))
    # lambda_f = 0 -> the fast window keeps only the last round's counts:
    # p = (k + a0) / (n + a0 + b0) = (8 + 1) / (32 + 2)
    assert abs(post.rate_fast - 9.0 / 34.0) < 1e-12
    assert post.rate_fast < post.rate_slow  # the slow window lags behind


def test_zero_prompt_round_decays_but_not_rounds_seen():
    tracker = make_tracker(lam=0.5, fast=0.5, slow=0.5)
    tracker.update(make_env_obs("e", n_prompts=4, group_size=8, success_rate=0.5))
    obs = make_env_obs("e", n_prompts=4, group_size=8, success_rate=0.5)
    decayed = obs.__class__(
        env_id="e", n_prompts=0, n_rollouts=0, k_success=0, n_groups_mixed=0,
        sum_score=0.0, sum_score_sq=0.0, prompt_tokens=0, completion_tokens=0,
        verifier_seconds=0.0, gpu_seconds=0.0, cost_usd=0.0,
    )
    post = tracker.update(decayed)
    assert post.n_rounds_seen == 1  # prompt rounds only
    assert post.n_groups_eff == 0.5 * 4.0  # counts decayed with no new data
    assert post.pass_rate > 0.25  # posterior keeps the prior + information


def test_envs_are_independent():
    tracker = make_tracker()
    tracker.update(make_env_obs("e1", n_prompts=4, group_size=8, success_rate=0.9))
    post = tracker.update(make_env_obs("e2", n_prompts=4, group_size=8, success_rate=0.1))
    assert post.env_id == "e2"
    assert post.pass_rate < 0.2
    assert tracker.rounds_seen("e1") == 1 and tracker.rounds_seen("e2") == 1


def test_checkpoint_round_trip():
    tracker = make_tracker()
    tracker.update(make_env_obs("e1", n_prompts=4, group_size=8, success_rate=0.5))
    tracker.update(make_env_obs("e2", n_prompts=2, group_size=8, success_rate=0.75))
    state = tracker.get_state()
    restored = make_tracker()
    restored.load_checkpoint(state)
    a = tracker.update(make_env_obs("e1", n_prompts=4, group_size=8, success_rate=0.6))
    b = restored.update(make_env_obs("e1", n_prompts=4, group_size=8, success_rate=0.6))
    assert a == b
