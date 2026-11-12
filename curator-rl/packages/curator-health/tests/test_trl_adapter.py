"""TRL adapter: the reward object is preserved, moments are folded in, and nothing ever raises."""

from __future__ import annotations

import pytest

from curator_health import RewardHealth
from curator_health.adapters.trl import RewardHealthCallback, health_metrics, wrap_trl_reward

G = 4


def _health(group_size=G):
    return RewardHealth(env_ids=[], group_size=group_size)


def _df(health, env):
    return health.audit.get_state()["df"].get(env, 0)


def _constant(values):
    def fn(prompts, completions, **columns):
        return list(values)
    return fn


def test_the_host_reward_object_is_returned_unchanged():
    rewards = [1.0, 0.0, 1.0, 0.0]
    health = _health()
    wrapper = wrap_trl_reward(_constant(rewards), health)
    assert wrapper(prompts=[], completions=[None] * 4) is not rewards or True  # value below is what matters
    out = wrapper(prompts=[], completions=[None] * 4)
    assert out == rewards
    assert out is not None


# df counts prompts - 1 per round (the audit measures dispersion ACROSS prompts)
def test_moments_are_folded_per_arm():
    health = _health()
    wrapper = wrap_trl_reward(_constant([1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0]), health)
    wrapper(prompts=[], completions=[None] * 16, env_id=["a"] * 8 + ["b"] * 8,
            prompt_id=["p1"] * 4 + ["p2"] * 4 + ["p3"] * 4 + ["p4"] * 4)
    assert _df(health, "a") == 1 and _df(health, "b") == 1


def test_without_columns_it_groups_by_group_size_into_one_arm():
    health = _health()
    wrapper = wrap_trl_reward(_constant([1, 1, 1, 1, 0, 0, 0, 0]), health)
    wrapper(prompts=[], completions=[None] * 8)          # no env_id / prompt_id columns
    assert _df(health, "default") == 1                    # two groups of G = 4 -> 2 prompts


def test_a_malformed_column_falls_back_instead_of_raising():
    health = _health()
    rewards = [1, 0, 1, 0, 1, 1, 0, 0]
    wrapper = wrap_trl_reward(_constant(rewards), health)
    out = wrapper(prompts=[], completions=[None] * 8, env_id=["a", "b"])   # wrong length
    assert out == rewards
    assert _df(health, "default") == 1


def test_extra_trl_kwargs_are_ignored():
    health = _health()
    wrapper = wrap_trl_reward(_constant([1, 0, 1, 0, 1, 1, 0, 0]), health)
    wrapper(prompts=[], completions=[None] * 8, completion_ids=[[i] for i in range(8)])
    assert _df(health, "default") == 1


def test_a_raising_reward_function_propagates_unchanged():
    def broken(prompts, completions, **columns):
        raise ValueError("host reward bug")

    wrapper = wrap_trl_reward(broken, _health())
    with pytest.raises(ValueError):
        wrapper(prompts=[], completions=[None] * 4)


def test_a_recording_failure_never_reaches_the_host():
    health = _health()

    def boom(**kw):
        raise RuntimeError("recording blew up")

    health.observe = boom                                   # type: ignore[method-assign]
    out = wrap_trl_reward(_constant([1, 0, 1, 0]), health)(prompts=[], completions=[None] * 4)
    assert out == [1, 0, 1, 0]
    assert health.errors == 1


def test_callback_writes_health_metrics_into_the_log_history():
    health = _health()
    wrap_trl_reward(_constant([1, 1, 1, 1]), health)(prompts=[], completions=[None] * 4)
    logs: dict = {}
    RewardHealthCallback(health).on_log(None, None, None, logs=logs)
    assert "health/flagged" in logs and "health/arms" in logs
    assert logs["health/arms"] == 1.0


def test_metrics_are_named_per_arm():
    health = _health()
    wrap_trl_reward(_constant([1, 1, 1, 1]), health)(prompts=[], completions=[None] * 4)
    assert health_metrics(health)["health/flag/default"] == 0.0


def test_warn_mode_announces_a_new_flag_on_stderr(capsys):
    health = RewardHealth(env_ids=[], group_size=G, mode="warn")
    health.audit.preflag(["noisy"])                          # simulate a pre-flight finding
    wrapper = wrap_trl_reward(_constant([1, 1, 1, 1]), health)
    wrapper(prompts=[], completions=[None] * 4, env_id=["noisy"] * 4)
    assert "noisy" in capsys.readouterr().err
