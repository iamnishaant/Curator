"""Unit tests for signal-richness (Roadmap E.3.3, Part M item 4).

Exactness tests pass lam=0.0 (no smoothing) so the stored value equals the
raw round value; the same convention lets the engine's tests work with small
exactly-checkable numbers. Signal outputs stay in [0, 1] by construction.
"""

from __future__ import annotations

import math
from dataclasses import replace

from tests.conftest import make_env_obs

from curator_rl.core.config import RichnessCfg
from curator_rl.signals.passrate import Posterior
from curator_rl.signals.richness import RichnessEstimator


def make_cfg(mode="mixed", band_lo=0.1, band_hi=0.9, variance_min=0.05) -> RichnessCfg:
    return RichnessCfg(mode=mode, band_lo=band_lo, band_hi=band_hi, variance_min=variance_min)


def test_mixed_mode_exact_first_round():
    est = RichnessEstimator(make_cfg("mixed"), lam=0.0, group_size=8)
    obs = make_env_obs("e", n_prompts=10, group_size=8, success_rate=0.5)
    assert est.update("e", replace(obs, n_groups_mixed=6), None) == 0.6  # type: ignore[arg-type]


def test_mixed_mode_then_smoothing_with_lam():
    est = RichnessEstimator(make_cfg("mixed"), lam=0.5, group_size=8)
    obs = make_env_obs("e", n_prompts=4, group_size=8, success_rate=0.5)
    # raw 1.0; first ew: 0.5*0 + 0.5*1
    assert abs(est.update("e", obs, None) - 0.5) < 1e-12  # type: ignore[arg-type]


def test_update_groups_exact_mixed_and_band():
    mixed = RichnessEstimator(make_cfg("mixed"), lam=0.0, group_size=4)
    rates = [0.0, 1.0, 0.3, 0.9]
    assert mixed.update_groups("e", rates) == 0.5  # 2 of 4 strictly inside (0,1)
    band = RichnessEstimator(make_cfg("band", band_lo=0.2, band_hi=0.8), lam=0.0, group_size=4)
    assert band.update_groups("e", rates) == 0.25  # only 0.3


def test_band_mode_posterior_overlap():
    est = RichnessEstimator(make_cfg("band", band_lo=0.1, band_hi=0.9), lam=0.0, group_size=8)
    post_full = _posterior(0.0, 1.0)   # full CI [0,1]: overlap 0.8 of span 1.0
    obs = make_env_obs("e", n_prompts=10, group_size=8, success_rate=0.5)
    assert abs(est.update("e", obs, post_full) - 0.8) < 1e-12  # type: ignore[arg-type]
    # a narrow CI inside the band: full coverage -> 1.0
    assert abs(est.update("e", obs, _posterior(0.3, 0.7)) - 1.0) < 1e-12  # type: ignore[arg-type]
    # a CI entirely below the band: zero overlap
    assert est.update("e", obs, _posterior(0.0, 0.05)) == 0.0  # type: ignore[arg-type]


def _posterior(lo: float, hi: float) -> Posterior:
    return Posterior(
        env_id="e", pass_rate=(lo + hi) / 2, pass_lo=lo, pass_hi=hi, se_group=0.0,
        n_groups_eff=10.0, n_rounds_seen=3, rate_fast=0.5, rate_slow=0.5,
        se_fast=0.0, se_slow=0.0,
    )


def test_variance_mode_std_threshold():
    est = RichnessEstimator(make_cfg("variance", variance_min=0.2), lam=0.0, group_size=4)
    # flat group rates -> zero variance -> richness 0
    assert est.update_groups("e", [0.5, 0.5, 0.5, 0.5]) == 0.0
    # spread rates: population var across groups = 0.25 -> std 0.5 > 0.2 -> 1
    rates = [0.0, 1.0, 0.8, 0.2]
    assert est.update_groups("e", rates) == 1.0


def test_zero_prompt_round_keeps_previous_value():
    est = RichnessEstimator(make_cfg("mixed"), lam=0.5, group_size=8)
    obs = make_env_obs("e", n_prompts=4, group_size=8, success_rate=0.5)
    est.update("e", obs, None)  # type: ignore[arg-type]
    empty = replace(obs, n_prompts=0, n_groups_mixed=0, sum_score=0.0, sum_score_sq=0.0, cost_usd=0.0)
    value = est.update("e", empty, None)  # type: ignore[arg-type]
    assert abs(value - 0.5) < 1e-12  # unchanged


def test_richness_bounded():
    est = RichnessEstimator(make_cfg("mixed"), lam=0.8, group_size=8)
    values = []
    for _ in range(20):
        obs = make_env_obs("e", n_prompts=8, group_size=8, success_rate=0.5)
        values.append(est.update("e", obs, None))  # type: ignore[arg-type]
    assert all(math.isclose(v, v) and 0.0 <= v <= 1.0 for v in values)
