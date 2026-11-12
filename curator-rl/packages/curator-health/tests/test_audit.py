"""In-run audit: a reward that ignores the prompt is caught; real verifiers are not.

Thresholds are the configured ones (d_max 1.5, z 2.33, min_df 40); populations are the measured ones
(verifier arms with prompt heterogeneity kappa ~ 1-2.5, dispersion D = 3.0-4.4; a prompt-independent
reward, D = 1).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from curator_health import HealthConfig, PromptDependenceAudit, RoundMoments

G = 8
CFG = HealthConfig(d_max=1.5, z=2.33, min_df=40)


def _round(rng, kappa, p, n):
    q = np.full(n, p) if kappa == math.inf else rng.beta(kappa * p, kappa * (1 - p), size=n)
    k = rng.binomial(G, q)
    r = k / G
    return RoundMoments("x", n, n * G, int(k.sum()), float(r.sum()), float((r * r).sum()))


def _first_flag(rng, kappa, p, rounds=60):
    audit = PromptDependenceAudit(CFG, ["x"], G)
    for t in range(rounds):
        if audit.observe(_round(rng, kappa, p, int(rng.integers(3, 11)))).flag:
            return t + 1
    return None


def test_dispersion_estimates_the_beta_binomial_ratio():
    rng = np.random.default_rng(0)
    for kappa, expected in ((math.inf, 1.0), (2.14, (G + 2.14) / 3.14)):
        audit = PromptDependenceAudit(CFG, ["x"], G)
        for _ in range(400):
            res = audit.observe(_round(rng, kappa, 0.35, 8))
        assert res.dispersion == pytest.approx(expected, rel=0.08)


def test_a_reward_that_ignores_the_prompt_is_flagged_early():
    rng = np.random.default_rng(1)
    firsts = [_first_flag(rng, math.inf, 0.35) for _ in range(200)]
    assert sum(f is not None for f in firsts) >= 196                 # >= 98% of 60-round runs
    assert np.median([f for f in firsts if f]) <= 14                 # typically by round ~11


def test_real_verifier_heterogeneity_is_never_flagged():
    rng = np.random.default_rng(2)
    for p in (0.1, 0.35, 0.6):
        firsts = [_first_flag(rng, 2.14, p) for _ in range(200)]
        assert sum(f is not None for f in firsts) == 0


def test_no_decision_before_min_df_and_uninformative_rounds_are_ignored():
    audit = PromptDependenceAudit(CFG, ["x"], G)
    zeros = RoundMoments("x", 8, 64, 0, 0.0, 0.0)                     # all-fail round: no information
    for _ in range(50):
        res = audit.observe(zeros)
    assert res.df == 0 and not res.flag and math.isnan(res.dispersion)
    rng = np.random.default_rng(3)
    for _ in range(4):                                                # 4 x 7 = 28 df < 40
        res = audit.observe(_round(rng, math.inf, 0.35, 8))
    assert res.df == 28 and not res.flag


def test_flag_clears_only_on_strong_evidence_of_prompt_dependence():
    rng = np.random.default_rng(4)
    audit = PromptDependenceAudit(CFG, ["x"], G)
    for _ in range(10):
        res = audit.observe(_round(rng, math.inf, 0.35, 8))
    assert res.flag
    for _ in range(3):
        res = audit.observe(_round(rng, 1.0, 0.35, 8))
    assert res.flag
    for _ in range(80):
        res = audit.observe(_round(rng, 1.0, 0.35, 8))
    assert not res.flag


def test_checkpoint_round_trip():
    rng = np.random.default_rng(5)
    a = PromptDependenceAudit(CFG, ["x"], G)
    for _ in range(7):
        a.observe(_round(rng, math.inf, 0.35, 8))
    b = PromptDependenceAudit(CFG, ["x"], G)
    b.load_checkpoint(a.get_state())
    obs = _round(rng, math.inf, 0.35, 8)
    assert a.observe(obs) == b.observe(obs)


def test_a_preflight_flag_holds_for_the_whole_run_and_survives_a_checkpoint():
    rng = np.random.default_rng(6)
    audit = PromptDependenceAudit(CFG, ["x", "y"], G)
    audit.preflag(["x", "not-an-arm"])
    assert audit.result("x").flag and not audit.result("y").flag
    for _ in range(80):                       # strong evidence of a real verifier never clears it
        audit.observe(_round(rng, 2.14, 0.35, 8))
    assert audit.result("x").flag
    clone = PromptDependenceAudit(CFG, ["x", "y"], G)
    clone.load_checkpoint(audit.get_state())
    assert clone.result("x").flag
    older = audit.get_state()
    del older["preflight"]                    # checkpoints from before the pre-flight feature
    clone.load_checkpoint(older)
    assert not clone.result("x").flag


def test_config_validation():
    with pytest.raises(ValueError):
        HealthConfig(d_max=1.0)
    with pytest.raises(ValueError):
        HealthConfig(z=0.0)
    with pytest.raises(ValueError):
        HealthConfig(min_df=1)
    with pytest.raises(ValueError):
        HealthConfig(flag_rate=1.5)
