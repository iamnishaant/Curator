"""The RewardHealth facade: modes, fail-open behaviour, reporting and checkpointing."""

from __future__ import annotations

import math

import pytest

from curator_health import HealthConfig, ProbePrompt, RewardHealth, RoundMoments

G = 8


def _moments(env_id="x", n=8, k=3):
    return RoundMoments(env_id, n, n * G, k, k / G * n, (k / G) ** 2 * n)


def test_shadow_mode_observes_but_does_not_act():
    health = RewardHealth(env_ids=["x"], group_size=G)
    assert health.mode == "shadow" and not health.acts
    health.observe(**_moments().__dict__)
    assert health.flags() == {"x": False}


def test_gate_mode_declares_that_the_host_should_act():
    health = RewardHealth(env_ids=["x"], group_size=G, mode="gate")
    assert health.acts


def test_observe_accepts_extra_keywords_and_never_raises():
    health = RewardHealth(env_ids=["x"], group_size=G)
    res = health.observe(env_id="x", n_prompts=8, n_rollouts=64, k_success=3,
                         sum_score=3.0, sum_score_sq=3.0, round=7)   # 'round' is ignored
    assert res.df == 7
    bad = health.observe(env_id="x")                                 # missing fields
    assert math.isnan(bad.dispersion) and health.errors == 1


def test_preflight_gate_flags_an_arm_for_the_whole_run():
    prompt = ProbePrompt(prompt_id="p", messages=({"role": "user", "content": "?"},), reference="42")
    leaky = lambda p, r: 1.0 if p.reference in r else 0.0        # noqa: E731
    health = RewardHealth(env_ids=["noisy"], group_size=G, config=HealthConfig(preflight_gate=True))
    report = health.preflight({"noisy": (leaky, [prompt])})
    assert report["flagged"] == ["noisy"]
    assert health.flags() == {"noisy": True}
    assert "noisy" in health.report()["flagged"]


def test_report_and_checkpoint_round_trip():
    health = RewardHealth(env_ids=["x"], group_size=G)
    for _ in range(5):
        health.observe(**_moments().__dict__)
    state = health.get_state()
    clone = RewardHealth(env_ids=["x"], group_size=G)
    clone.load_checkpoint(state)
    assert clone.flags() == health.flags()
    assert clone.report()["mode"] == "shadow"


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError):
        RewardHealth(env_ids=["x"], group_size=G, mode="nonsense")
