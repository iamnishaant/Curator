"""Unit tests for the Calibrated Reward Engine (D-76)."""

from __future__ import annotations

import math

import pytest
from tests.conftest import BASE_CFG

from curator_rl.calibration.cre import CalibratedRewardEngine
from curator_rl.core.config import load_config

BUDGET = 10.0


def make(mode="full", rho=1.0, roi_scale=0.2, discount=1.0, scale=0.015, envs=("a", "b", "c")):
    cfg = load_config(BASE_CFG).cre.model_copy(update={
        "enabled": True, "mode": mode, "prior_rel_sd": rho, "roi_scale": roi_scale, "discount": discount,
        "proxy_scale": scale})
    return CalibratedRewardEngine(envs, cfg)


def gain(g, se, usd):
    return {"gain": g, "se": se, "usd": usd, "share": 0.5}


PX = {"a": 0.5, "b": 0.5, "c": 0.5}
UC = {"a": 0.5, "b": 0.5, "c": 0.5}


def two_evidence_arms(cre, *, ready=True):
    # a: gain 0.04 over 2 dollars, se 0.01 => num 800, prec 4e4, b_hat 0.02
    # b: gain 0.02 over 2 dollars => b_hat 0.01 ; online ratios b*c/x are 0.02 and 0.01 (diagnostic)
    cre.ingest({"a": gain(0.04, 0.01, 2.0), "b": gain(0.02, 0.01, 2.0)}, PX, UC, ready=ready, window_k=1)


def test_likelihood_accumulation_and_scale_hand_computed():
    cre = make()
    two_evidence_arms(cre)
    assert cre.b_hat("a") == pytest.approx(0.02)
    assert cre.b_hat("b") == pytest.approx(0.01)
    assert cre.b_hat("c") is None
    assert cre.online_scale == pytest.approx(0.015)   # median of {0.02, 0.01}: diagnostic only
    assert cre.scale == 0.015                         # the frozen constant drives rewards
    assert cre.active


def test_posterior_mean_is_precision_weighted_shrinkage_toward_the_proxy_prior():
    cre = make()
    two_evidence_arms(cre)
    est = cre.estimate("a", 0.5, 0.5, BUDGET)
    # m0 = 0.015*0.5/0.5 = 0.015 ; v0 = 0.015^2 ; data: prec 4e4, num 800
    assert est.prior_mean == pytest.approx(0.015)
    assert est.mean_per_usd == pytest.approx((0.015 / 0.015**2 + 800.0) / (1 / 0.015**2 + 4e4))
    assert 0.015 < est.mean_per_usd < 0.02            # between the prior and the evidence
    assert est.sd_per_usd == pytest.approx(1.0 / math.sqrt(1 / 0.015**2 + 4e4))
    assert est.source == "posterior" and est.n_obs == 1
    # reward = clip(mean * B / R_max)
    assert est.reward == pytest.approx(min(est.mean_per_usd * BUDGET / 0.2, 1.0))


def test_arm_without_evidence_sits_exactly_at_its_proxy_prior():
    cre = make()
    two_evidence_arms(cre)
    est = cre.estimate("c", 0.5, 0.5, BUDGET)
    assert est.source == "prior" and est.n_obs == 0
    assert est.mean_per_usd == pytest.approx(est.prior_mean) == pytest.approx(0.015)
    assert est.reward == pytest.approx(0.015 * BUDGET / 0.2)    # 0.75


def test_no_uncertainty_ablation_uses_the_point_estimate():
    cre = make(mode="no_uncertainty")
    two_evidence_arms(cre)
    assert cre.estimate("a", 0.5, 0.5, BUDGET).mean_per_usd == pytest.approx(0.02)     # b_hat, no shrinkage
    assert cre.estimate("c", 0.5, 0.5, BUDGET).mean_per_usd == pytest.approx(0.015)    # prior mean


def test_more_evidence_means_a_smaller_sd_and_a_pull_toward_the_data():
    cre = make()
    two_evidence_arms(cre)
    sd1 = cre.estimate("a", 0.5, 0.5, BUDGET).sd_per_usd
    cre.ingest({"a": gain(0.04, 0.01, 2.0)}, PX, UC, ready=True, window_k=2)
    est2 = cre.estimate("a", 0.5, 0.5, BUDGET)
    assert est2.sd_per_usd < sd1 and est2.n_obs == 2


def test_discounting_forgets_old_evidence_per_window():
    cre = make(discount=0.5)
    cre.ingest({"a": gain(0.04, 0.01, 2.0)}, PX, UC, ready=True, window_k=1)
    prec1 = 4.0 / 1e-4
    cre.ingest({"a": gain(0.04, 0.01, 2.0)}, PX, UC, ready=True, window_k=2)
    assert cre.get_state()["prec"]["a"] == pytest.approx(0.5 * prec1 + prec1)


def test_reward_scale_is_the_frozen_constant_not_the_online_estimate():
    """D-78: a junk arm among the few evaluated arms must not drag the prior scale to ~0."""
    cre = make(scale=0.05)
    cre.ingest({"a": gain(0.0, 0.01, 2.0), "b": gain(0.001, 0.01, 2.0)}, PX, UC, ready=True, window_k=1)
    assert cre.online_scale < 0.001                    # the fragile online median collapsed
    est = cre.estimate("c", 0.5, 0.5, BUDGET)          # arm without evidence
    assert est.prior_mean == pytest.approx(0.05)       # still the frozen scale * x / c


def test_not_active_before_ready_or_without_evidence():
    cre = make()
    two_evidence_arms(cre, ready=False)
    assert not cre.active
    empty = make()
    empty.ingest({}, PX, UC, ready=True, window_k=1)
    assert not empty.active


def test_reward_is_clipped_and_negative_gain_gives_zero():
    cre = make()
    cre.ingest({"a": gain(-0.20, 0.01, 2.0), "b": gain(0.02, 0.01, 2.0)}, PX, UC, ready=True, window_k=1)
    assert cre.estimate("a", 0.5, 0.5, BUDGET).reward == 0.0          # decisively negative evidence
    big = make(roi_scale=0.001)
    two_evidence_arms(big)
    assert big.estimate("a", 0.5, 0.5, BUDGET).reward == 1.0          # saturates, never exceeds 1


def test_extreme_costs_do_not_blow_up():
    cre = make()
    two_evidence_arms(cre)
    for c in (0.0, 1e-15, 1e9):
        est = cre.estimate("c", 0.5, c, BUDGET)
        assert 0.0 <= est.reward <= 1.0 and math.isfinite(est.mean_per_usd) and math.isfinite(est.sd_per_usd)
    assert cre.estimate("c", 0.5, 0.0, BUDGET).prior_mean == 0.0       # unknown cost: no inflated prior
    assert cre.estimate("c", 0.0, 0.5, 0.0).reward >= 0.0              # zero budget guarded


def test_zero_dollar_spans_are_ignored():
    cre = make()
    cre.ingest({"a": gain(0.05, 0.01, 0.0)}, PX, UC, ready=True, window_k=1)
    assert cre.b_hat("a") is None and not cre.active


def test_reward_of_one_arm_does_not_depend_on_another_arms_current_signal():
    cre = make()
    two_evidence_arms(cre)
    r1 = cre.estimate("a", 0.5, 0.5, BUDGET)
    cre.estimate("b", 99.0, 0.5, BUDGET)                               # another arm's extreme signal
    assert cre.estimate("a", 0.5, 0.5, BUDGET) == r1                   # per-arm fixed transform


def test_checkpoint_roundtrip_reproduces_estimates():
    a = make()
    two_evidence_arms(a)
    b = make()
    b.load_checkpoint(a.get_state())
    assert b.active and b.online_scale == a.online_scale
    assert b.estimate("a", 0.4, 0.6, BUDGET) == a.estimate("a", 0.4, 0.6, BUDGET)
