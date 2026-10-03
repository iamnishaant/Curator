"""Unit tests for weight-to-quota conversion (Roadmap D.4)."""

import pytest

from curator_rl.simulator.quotas import largest_remainder_quotas


def test_exact_proportions():
    q = largest_remainder_quotas({"a": 0.5, "b": 0.5}, 80)
    assert q == {"a": 40, "b": 40}


def test_sum_is_exact_with_remainder_rounding():
    q = largest_remainder_quotas({"a": 0.2, "b": 0.3, "c": 0.5}, 80)
    assert sum(q.values()) == 80
    assert q["a"] == 16 and q["b"] == 24 and q["c"] == 40


def test_awkward_fractions_still_sum_to_total():
    q = largest_remainder_quotas({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, 100)
    assert sum(q.values()) == 100
    assert all(abs(v - 33.33) < 1 for v in q.values())


def test_deterministic_tie_break():
    q1 = largest_remainder_quotas({"a": 0.15, "b": 0.15, "c": 0.7}, 10, tie_break_seed=3)
    q2 = largest_remainder_quotas({"a": 0.15, "b": 0.15, "c": 0.7}, 10, tie_break_seed=3)
    assert q1 == q2
    assert sum(q1.values()) == 10


def test_zero_and_negative_totals():
    assert largest_remainder_quotas({"a": 1.0}, 0) == {"a": 0}
    with pytest.raises(ValueError):
        largest_remainder_quotas({"a": 1.0}, -1)


def test_empty_weights():
    assert largest_remainder_quotas({}, 10) == {}
