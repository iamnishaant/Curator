"""Unit tests for the GRPO group advantage statistic (D-68)."""

import numpy as np
import pytest

from curator_rl.core.advantage import GRPO_STD_EPS, group_mean_abs_advantage


def _explicit(k: int, g: int) -> float:
    """TRL-style advantages: (r - mean) / (unbiased std + 1e-4), averaged in |.|."""
    rewards = np.array([1.0] * k + [0.0] * (g - k))
    adv = (rewards - rewards.mean()) / (rewards.std(ddof=1) + GRPO_STD_EPS)
    return float(np.abs(adv).mean())


@pytest.mark.parametrize("g", [2, 4, 8, 16])
def test_closed_form_matches_explicit_advantages(g):
    for k in range(1, g):
        assert group_mean_abs_advantage(k, g) == pytest.approx(_explicit(k, g), rel=1e-12)


@pytest.mark.parametrize("g", [4, 8])
def test_zero_variance_groups_have_zero_advantage(g):
    assert group_mean_abs_advantage(0, g) == 0.0
    assert group_mean_abs_advantage(g, g) == 0.0


def test_symmetric_and_peaks_at_half():
    g = 8
    vals = [group_mean_abs_advantage(k, g) for k in range(g + 1)]
    assert vals == pytest.approx(vals[::-1])           # k <-> g-k
    assert max(vals) == vals[g // 2]


def test_hand_computed_g8_k2():
    # p = 0.25, unbiased std = sqrt(8/7 * 0.1875) = 0.46291 ; 2*0.25*0.75/(0.46291+1e-4)
    assert group_mean_abs_advantage(2, 8) == pytest.approx(0.375 / (0.462910 + 1e-4), rel=1e-4)


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        group_mean_abs_advantage(1, 1)
    with pytest.raises(ValueError):
        group_mean_abs_advantage(9, 8)
    with pytest.raises(ValueError):
        group_mean_abs_advantage(-1, 8)
