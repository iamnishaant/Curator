"""Unit tests for window gains, credit rules and the calibrator (Roadmap Part I, D-72/D-73)."""

from __future__ import annotations

import math

import pytest
from tests.conftest import BASE_CFG

from curator_rl.calibration.calibrator import Calibrator
from curator_rl.calibration.credit import (
    OwnSliceCredit,
    exposure_shares,
    share_credit,
    window_deltas,
)
from curator_rl.core.config import load_config
from curator_rl.core.types import CalibrationObservation


def cobs(k, scores, *, se=0.05, exposure=None, paired=None):
    return CalibrationObservation(
        window_k=k, round=5 * k, score_total=sum(scores.values()) / len(scores),
        score_by_domain=dict(scores), se_total=se, se_by_domain={d: se for d in scores},
        n_items=200 * len(scores), eval_cost_usd=0.0,
        exposure_by_env=dict(exposure or {d: 1.0 for d in scores}), window_cost_usd=1.0,
        delta_se_by_domain=paired,
    )


def test_window_deltas_independent_and_paired_se():
    a = cobs(1, {"x": 0.40, "y": 0.50}, se=0.03)
    b = cobs(2, {"x": 0.45, "y": 0.48}, se=0.04)
    delta, se = window_deltas(a, b)
    assert delta == pytest.approx({"x": 0.05, "y": -0.02})
    assert se["x"] == pytest.approx(0.05)                     # hypot(0.04, 0.03)
    c = cobs(2, {"x": 0.45, "y": 0.48}, paired={"x": 0.01, "y": 0.02})
    _, se_p = window_deltas(a, c)
    assert se_p == pytest.approx({"x": 0.01, "y": 0.02})     # paired SE wins


def test_shares_and_c1_credit():
    shares = exposure_shares({"a": 30.0, "b": 10.0, "c": 0.0})
    assert shares == pytest.approx({"a": 0.75, "b": 0.25, "c": 0.0})
    assert share_credit(0.04, {"a": 30.0, "b": 10.0, "c": 0.0}) == pytest.approx({"a": 0.03, "b": 0.01, "c": 0.0})
    assert exposure_shares({"a": 0.0}) == {"a": 0.0}


def test_own_slice_credit_weighted_least_squares_hand_computed():
    cr = OwnSliceCredit(["a"])
    cr.add("a", 0.5, 0.02, 0.01)   # W=0.5, y=0.02, SE=0.01 -> w=1e4
    cr.add("a", 1.0, 0.05, 0.02)   # W=1.0, y=0.05, SE=0.02 -> w=2500
    # b = sum(W y w) / sum(W^2 w) = (0.5*0.02*1e4 + 1*0.05*2500) / (0.25*1e4 + 1*2500) = 225/5000
    mean, sd = cr.estimate("a")
    assert mean == pytest.approx(0.045)
    assert sd == pytest.approx(1.0 / math.sqrt(5000.0))
    cr.add("a", 0.0, 9.9, 0.01)    # zero exposure contributes nothing
    assert cr.windows("a") == 2
    # a prior N(0, 0.01^2) adds precision 1e4 and pulls toward 0: 225 / 15000
    assert cr.estimate("a", prior_mean=0.0, prior_sd=0.01)[0] == pytest.approx(0.015)


def test_own_slice_credit_without_data_returns_prior():
    cr = OwnSliceCredit(["a"])
    assert cr.estimate("a") == (0.0, float("inf"))
    cr2 = OwnSliceCredit(["a"])
    cr2.load_checkpoint(cr.get_state())
    assert cr2.windows("a") == 0


def _calibrator(k_min=2, z_mis=2.0):
    cfg = load_config(BASE_CFG).calib.model_copy(update={"k_min": k_min, "z_mis": z_mis})
    return Calibrator(["good", "junk", "nodomain"], cfg, domain_of={"nodomain": None})


def _run(cal, windows):
    """windows: list of (scores, exposure, proxy_x); returns the reports."""
    out = []
    for k, (scores, exposure, px) in enumerate(windows, start=1):
        out.append(cal.observe(cobs(k, scores, exposure=exposure, paired={d: 0.005 for d in scores}), px))
    return out


def test_calibrator_flags_the_arm_whose_proxy_overpromises():
    # 'good' and 'junk' get equal compute and equal proxy, but only 'good''s slice rises
    exposure = {"good": 40.0, "junk": 40.0, "nodomain": 20.0}
    px = {"good": 0.6, "junk": 0.6, "nodomain": 0.6}
    windows = []
    for k in range(5):
        windows.append(({"good": 0.30 + 0.02 * k, "junk": 0.30}, exposure, px))
    reports = _run(_calibrator(), windows)
    assert not reports[0].ready and reports[0].mismatch == {}       # first eval is the baseline
    last = reports[-1]
    assert last.ready
    assert last.gain_hat["good"] == pytest.approx(0.02 / 0.4)        # 0.02 per window at share 0.4
    assert last.gain_hat["junk"] == pytest.approx(0.0, abs=1e-12)
    assert last.mismatch == {"good": False, "junk": True, "nodomain": False}
    assert last.z["junk"] < -2.0


def test_calibrator_scale_is_robust_median_and_never_flags_underestimation():
    exposure = {"good": 50.0, "junk": 50.0, "nodomain": 0.0}
    px = {"good": 0.1, "junk": 0.9, "nodomain": 0.0}   # proxy UNDER-rates 'good'
    windows = [({"good": 0.30 + 0.03 * k, "junk": 0.30}, exposure, px) for k in range(5)]
    last = _run(_calibrator(), windows)[-1]
    assert last.z["good"] >= 0.0 and not last.mismatch["good"]    # underestimation is never flagged


def test_no_flags_before_k_min_windows():
    exposure = {"good": 40.0, "junk": 40.0, "nodomain": 20.0}
    px = {"good": 0.6, "junk": 0.6, "nodomain": 0.6}
    windows = [({"good": 0.30 + 0.02 * k, "junk": 0.30}, exposure, px) for k in range(3)]
    reports = _run(_calibrator(k_min=5), windows)
    assert all(not r.ready and not any(r.mismatch.values()) for r in reports)


def test_calibrator_checkpoint_roundtrip_reproduces_next_report():
    exposure = {"good": 40.0, "junk": 40.0, "nodomain": 20.0}
    px = {"good": 0.6, "junk": 0.6, "nodomain": 0.6}
    windows = [({"good": 0.30 + 0.02 * k, "junk": 0.30}, exposure, px) for k in range(4)]
    a = _calibrator()
    _run(a, windows)
    b = _calibrator()
    b.load_checkpoint(a.get_state())
    nxt = cobs(5, {"good": 0.40, "junk": 0.30}, exposure=exposure, paired={"good": 0.005, "junk": 0.005})
    ra, rb = a.observe(nxt, px), b.observe(nxt, px)
    assert ra == rb


def _targeting_calibrator(max_targets=1):
    cfg = load_config(BASE_CFG).calib.model_copy(update={
        "k_min": 1, "targeting": "exposure", "max_targets": max_targets, "items_per_slice": 50})
    return Calibrator(["a", "b", "c"], cfg)


def test_targeting_baseline_evaluates_every_slice_then_the_most_funded():
    cal = _targeting_calibrator(max_targets=2)
    assert cal.select_targets({"a": 0.1, "b": 0.5, "c": 0.4}) == {"a": 50, "b": 50, "c": 50}  # baseline
    cal.observe(cobs(1, {"a": 0.3, "b": 0.3, "c": 0.3}), {"a": 0.5, "b": 0.5, "c": 0.5})
    assert cal.select_targets({"a": 0.1, "b": 0.5, "c": 0.4}) == {"b": 50, "c": 50}
    assert cal.select_targets({"a": 0.3, "b": 0.3, "c": 0.3}) == {"a": 50, "b": 50}   # ties: env id


def test_targeting_all_mode_always_requests_every_slice():
    cfg = load_config(BASE_CFG).calib.model_copy(update={"targeting": "all", "items_per_slice": 30})
    cal = Calibrator(["a", "b"], cfg)
    cal.observe(cobs(1, {"a": 0.3, "b": 0.3}), {"a": 0.5, "b": 0.5})
    assert cal.select_targets({"a": 1.0, "b": 0.0}) == {"a": 30, "b": 30}


def test_span_gain_uses_compute_since_the_slice_was_last_evaluated():
    """A slice skipped for a window contributes one gain over the whole span."""
    cal = _targeting_calibrator()
    exp = {"a": 50.0, "b": 50.0, "c": 0.0}
    px = {"a": 0.5, "b": 0.5, "c": 0.5}
    cal.observe(cobs(1, {"a": 0.30, "b": 0.30, "c": 0.30}, exposure=exp), px)          # baseline
    cal.observe(cobs(2, {"b": 0.30}, exposure=exp, paired={"b": 0.01}), px)            # 'a' skipped
    rep = cal.observe(cobs(3, {"a": 0.40, "b": 0.30}, exposure=exp,
                           paired={"a": 0.01, "b": 0.01}), px)
    # 'a' got share 0.5 in each of windows 2 and 3 -> span share 1.0, gain 0.10 -> b_a = 0.10
    assert rep.gain_hat["a"] == pytest.approx(0.10)
    assert rep.evaluated == ("a", "b")
    assert rep.delta_by_domain["a"] == pytest.approx(0.10)
