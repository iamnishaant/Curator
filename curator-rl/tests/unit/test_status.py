"""Unit tests for the S1-S5 status classifier (Roadmap E.5, Part M item 9)."""

from __future__ import annotations

from curator_rl.core.config import StatusCfg
from curator_rl.core.types import EnvStatus
from curator_rl.signals.status import StatusClassifier, flip_rate


def make_cfg(**kwargs) -> StatusCfg:
    base = dict(
        n_min=8, r_min=2, p_sat=0.8, p_hard=0.1, sr_hard=0.1, z_up=2.0, z_neg=2.0,
        hysteresis_p=0.05, hysteresis_sr=0.05, dwell_min=3, consecutive_rounds=2,
        s1_entry_ratio=0.5,
    )
    base.update(kwargs)
    return StatusCfg(**base)


def step(cl: StatusClassifier, *, p=0.5, lo=None, hi=None, z=0.0, sr=1.0,
         groups=100.0, seen=10) -> EnvStatus:
    status, _ = cl.step(
        n_groups_eff=groups,
        rounds_seen=seen,
        p_hat=p,
        p_lo=p if lo is None else lo,
        p_hi=p if hi is None else hi,
        z_lp=z,
        sr=sr,
        )
    return status


# S1 --------------------------------------------------------------------------

def test_s1_stays_until_both_minimums_met():
    cl = StatusClassifier(make_cfg(), mismatch_windows=2, clear_windows=2)
    assert step(cl, p=0.5, groups=4.0) == EnvStatus.S1  # groups < n_min
    assert step(cl, p=0.5, groups=100.0, seen=1) == EnvStatus.S1  # seen < r_min
    assert step(cl, p=0.5) == EnvStatus.S2           # both minimums met -> leaves


def test_s1_has_priority_over_mismatch():
    cl = StatusClassifier(make_cfg(), mismatch_windows=1, clear_windows=2)
    cl.set_mismatch(True)                       # streak 1 >= q=1 -> S5 condition hot
    assert step(cl, p=0.5, seen=1) == EnvStatus.S1  # but S1 wins (priority)


def test_s1_reentry_uses_the_hysteresis_floor():
    # n_min=8, entry ratio 0.5 -> re-entry needs evidence < 4; exit needs >= 8
    cl = StatusClassifier(make_cfg(), mismatch_windows=2, clear_windows=2)
    assert step(cl, p=0.5, seen=1, groups=8.0) == EnvStatus.S1   # first: evidence low
    assert step(cl, p=0.5, groups=8.0) == EnvStatus.S2           # exit at n_min
    assert step(cl, p=0.5, groups=5.0) == EnvStatus.S2           # 5 >= floor(4): no re-entry
    assert step(cl, p=0.5, groups=3.9) == EnvStatus.S1           # decayed below the floor
    assert step(cl, p=0.5, groups=6.0) == EnvStatus.S1           # still < n_min inside S1
    assert step(cl, p=0.5, groups=8.0) == EnvStatus.S2           # evidence restored


# S3 --------------------------------------------------------------------------

def test_s3_entry_needs_consecutive_rounds():
    cl = StatusClassifier(make_cfg(consecutive_rounds=2), mismatch_windows=2, clear_windows=2)
    ok = dict(p=0.9, lo=0.88, hi=0.92, z=0.5, sr=0.0)   # s3 condition true
    miss = dict(p=0.5, lo=0.5, hi=0.5, z=0.0, sr=0.0)   # condition false
    assert step(cl, **ok, seen=1) == EnvStatus.S1       # S1 still (r_min)
    assert step(cl, **miss) == EnvStatus.S2             # run reset by the miss
    assert step(cl, **ok) == EnvStatus.S2               # run=1 < h=2 -> no entry
    assert step(cl, **ok) == EnvStatus.S3               # run=2 -> enters


def test_s3_oscillation_inside_hysteresis_yields_no_flips():
    # the roadmap's named anti-oscillation test: p_hat oscillates across the
    # saturation boundary inside the hysteresis band -> zero transitions
    cl = StatusClassifier(make_cfg(consecutive_rounds=2, dwell_min=3),
                          mismatch_windows=2, clear_windows=2)
    args = dict(p=0.9, lo=0.88, hi=0.92, z=0.5, sr=0.0)
    step(cl, p=0.5, seen=1)
    step(cl, **args)
    step(cl, **args)
    assert cl.status == EnvStatus.S3
    trace = [cl.status]
    # oscillation STRICTLY inside the hysteresis band: p_sat - hys = 0.75 <
    # p_hat < p_sat = 0.8, while the entry condition (p_lo >= 0.8) stays false
    for p_hat in (0.76, 0.79, 0.76, 0.79, 0.76, 0.79):
        trace.append(step(cl, p=p_hat, lo=0.72, hi=0.80, z=0.0, sr=0.0))
    assert set(trace) == {EnvStatus.S3}  # hysteresis + dwell pin it
    assert flip_rate(trace) == 0.0


def test_established_s3_is_not_demoted_to_s1_by_evidence_decay():
    # D-51: a saturated arm that gets starved (n_groups_eff decays below the S1
    # floor) stays S3 -- evidence decay alone must not send it back to S1.
    cl = StatusClassifier(make_cfg(consecutive_rounds=2), mismatch_windows=2, clear_windows=2)
    ok = dict(p=0.9, lo=0.88, hi=0.92, z=0.5, sr=0.0)
    step(cl, p=0.5, seen=1)
    step(cl, **ok)
    step(cl, **ok)
    assert cl.status == EnvStatus.S3
    assert step(cl, **ok, groups=1.0) == EnvStatus.S3   # 1 << floor (n_min*0.5 = 4)


def test_established_s4_is_not_demoted_to_s1_by_evidence_decay():
    cl = StatusClassifier(make_cfg(consecutive_rounds=2, dwell_min=2), mismatch_windows=2, clear_windows=2)
    args = dict(p=0.05, lo=0.03, hi=0.08, z=0.0, sr=0.05)
    step(cl, p=0.5, seen=1)
    step(cl, **args)
    step(cl, **args)
    assert cl.status == EnvStatus.S4
    assert step(cl, **args, groups=0.5) == EnvStatus.S4  # starved too-hard arm stays S4


def test_s5_overrides_an_established_s3():
    # mismatch is enter-exempt from anywhere, including a saturated incumbent
    cl = StatusClassifier(make_cfg(consecutive_rounds=2), mismatch_windows=1, clear_windows=2)
    ok = dict(p=0.9, lo=0.88, hi=0.92, z=0.5, sr=0.0)
    step(cl, p=0.5, seen=1)
    step(cl, **ok)
    step(cl, **ok)
    assert cl.status == EnvStatus.S3
    cl.set_mismatch(True)
    assert step(cl, **ok) == EnvStatus.S5


def test_s2_arm_still_reenters_s1_when_evidence_decays():
    # the demotion exemption is for S3/S4 incumbents only; S2 keeps the E.5 rule
    cl = StatusClassifier(make_cfg(), mismatch_windows=2, clear_windows=2)
    step(cl, p=0.5, seen=1, groups=8.0)
    assert step(cl, p=0.5, groups=8.0) == EnvStatus.S2
    assert step(cl, p=0.5, groups=3.9) == EnvStatus.S1


# S4 --------------------------------------------------------------------------

def test_s4_entry_and_hysteresis_leave():
    cl = StatusClassifier(make_cfg(consecutive_rounds=2, dwell_min=2), mismatch_windows=2, clear_windows=2)
    args = dict(p=0.05, lo=0.03, hi=0.08, z=0.0, sr=0.05)  # too-hard condition
    step(cl, p=0.5, seen=1)
    step(cl, **args)                       # S2, run 1
    assert step(cl, **args) == EnvStatus.S4  # run 2 >= h
    # recovery above p_hard + hys (0.15)
    assert step(cl, p=0.2, lo=0.18, hi=0.22, z=0.0, sr=1.0) == EnvStatus.S4  # dwell=2 holds it
    assert step(cl, p=0.2, lo=0.18, hi=0.22, z=0.0, sr=1.0) == EnvStatus.S2  # dwell passed: leaves


def test_s4_needs_low_richness():
    cl = StatusClassifier(make_cfg(consecutive_rounds=2), mismatch_windows=2, clear_windows=2)
    args = dict(p=0.05, lo=0.03, hi=0.08, z=0.0, sr=0.9)  # sr too high -> not S4
    step(cl, p=0.5, seen=1)
    step(cl, **args)
    assert step(cl, **args) == EnvStatus.S2


# S5 --------------------------------------------------------------------------

def test_s5_mismatch_streaks_enter_and_clear():
    cl = StatusClassifier(make_cfg(), mismatch_windows=2, clear_windows=2)
    cl.set_mismatch(True)                       # streak 1
    assert step(cl, p=0.5) == EnvStatus.S2      # below q -> no S5
    cl.set_mismatch(True)                       # streak 2 >= q -> S5 enters immediately
    assert step(cl, p=0.5) == EnvStatus.S5
    cl.set_mismatch(False)                      # cold streak 1
    assert step(cl, p=0.5) == EnvStatus.S5      # < q' -> holds
    cl.set_mismatch(False)                      # cold streak 2 >= q' -> leaves
    assert step(cl, p=0.5) == EnvStatus.S2


def test_s5_entering_ignores_dwell():
    cl = StatusClassifier(make_cfg(dwell_min=20), mismatch_windows=1, clear_windows=5)
    assert step(cl, p=0.5, seen=1) == EnvStatus.S1
    cl.set_mismatch(True)
    assert step(cl, p=0.5) == EnvStatus.S5  # entering S5 exempt even from long dwell


# flip_rate --------------------------------------------------------------------

def test_flip_rate_metric():
    trace = [EnvStatus.S1, EnvStatus.S1, EnvStatus.S2, EnvStatus.S2, EnvStatus.S3]
    assert abs(flip_rate(trace) - 2 * 100.0 / 4) < 1e-12
    assert flip_rate([]) == 0.0
    assert flip_rate([EnvStatus.S2]) == 0.0
    warm = [EnvStatus.S1] + trace
    assert abs(flip_rate(warm, warmup=1) - flip_rate(trace)) < 1e-12
