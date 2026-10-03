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
