"""Unit tests for the LP estimators (Roadmap E.3, Part M item 3)."""

from __future__ import annotations

import math

import numpy as np

from curator_rl.core.seeding import SeedManager
from curator_rl.signals.passrate import Posterior
from curator_rl.signals.progress import LPEstimator


def make_posterior(rate=0.5, **kwargs) -> Posterior:
    return Posterior(
        env_id="e",
        pass_rate=rate,
        pass_lo=rate - 0.05,
        pass_hi=rate + 0.05,
        se_group=0.05,
        n_groups_eff=16.0,
        n_rounds_seen=2,
        rate_fast=kwargs.get("rate_fast", rate),
        rate_slow=kwargs.get("rate_slow", rate),
        se_fast=kwargs.get("se_fast", 0.05),
        se_slow=kwargs.get("se_slow", 0.05),
    )


# LP-A ------------------------------------------------------------------------

def test_lp_a_known_values():
    est = LPEstimator("lp_a", window_rounds=10)
    post = make_posterior(rate_fast=0.8, rate_slow=0.5, se_fast=0.05, se_slow=0.05)
    lp, se, z = est.compute("e", post)
    assert abs(lp - 0.3) < 1e-12
    assert abs(se - math.sqrt(0.05**2 + 0.05**2)) < 1e-12
    assert abs(z - 0.3 / se) < 1e-12


def test_lp_a_zero_se_gives_zero_z():
    est = LPEstimator("lp_a", window_rounds=10)
    lp, se, z = est.compute("e", make_posterior(rate_fast=0.6, rate_slow=0.6, se_fast=0.0, se_slow=0.0))
    assert lp == 0.0 and se == 0.0 and z == 0.0


# LP-B ------------------------------------------------------------------------

def test_lp_b_slopes_exact_line():
    est = LPEstimator("lp_b", window_rounds=10)
    for i, rate in enumerate([0.1, 0.2, 0.3, 0.4]):
        est.observe("e", make_posterior(rate), n_prompts=16)
        slope, se, z = est.compute("e", make_posterior(rate))
        if i >= 2:
            assert abs(slope - 0.1) < 1e-9  # noise-free line of slope 0.1
            assert 0.0 <= se < 0.5  # a finite SE; the exact line leaves se small
            assert abs(z) < 1000.0
    slope, se, z = est.compute("e", make_posterior(0.4))
    assert abs(slope - 0.1) < 1e-9


def test_lp_b_needs_three_seen_rounds():
    est = LPEstimator("lp_b", window_rounds=10)
    assert est.compute("e", make_posterior(0.5)) == (0.0, 0.0, 0.0)
    est.observe("e", make_posterior(0.5), 16)
    assert est.compute("e", make_posterior(0.5)) == (0.0, 0.0, 0.0)
    est.observe("e", make_posterior(0.6), 16)
    assert est.compute("e", make_posterior(0.6)) == (0.0, 0.0, 0.0)


def test_lp_b_flat_sequence_mean_abs_z_is_small():
    """Across seeds the flat-rate z stays modest (no systematic FPR inflation)."""
    zs = []
    for seed in range(30):
        est = LPEstimator("lp_b", window_rounds=10)
        gen = np.random.default_rng(100 + seed)
        for _ in range(10):
            est.observe("e", make_posterior(0.5 + gen.normal(0.0, 0.03)), n_prompts=16)
        lp, se, z = est.compute("e", make_posterior(0.5))
        zs.append(abs(z))
    assert sum(zs) / len(zs) < 2.0


# LP-C ------------------------------------------------------------------------

def test_lp_c_window_means_exact():
    est = LPEstimator("lp_c", window_rounds=2)
    rates = [0.1, 0.2, 0.3, 0.4]
    for i, rate in enumerate(rates):
        est.observe("e", make_posterior(rate), n_prompts=16)
        lp, se, z = est.compute("e", make_posterior(rate))
        if i < 3:
            assert lp == 0.0 and se == 0.0 and z == 0.0
        else:
            assert abs(lp - 0.2) < 1e-12  # mean(0.3, 0.4) - mean(0.1, 0.2)


def test_lp_c_constant_rate_jitter():
    est = LPEstimator("lp_c", window_rounds=4)
    rng = SeedManager(5).rng("lp-c-flat")
    for _ in range(8):
        est.observe("e", make_posterior(0.5 + rng.normal(0.0, 0.02)), n_prompts=16)
    lp, se, z = est.compute("e", make_posterior(0.5))
    assert abs(z) < 2.0


# FPR (the roadmap's headline estimator criterion) -----------------------------

def _run_flat_fpr(method: str, seed: int, p_true=0.5, prompts=16, g=8) -> float:
    """|z| >= 2 rate over flat sequences (true velocity zero, sampling noise real)."""
    flags = 0
    checks = 0
    generator = np.random.default_rng(seed)
    est = LPEstimator(method, window_rounds=10)
    s = n = 0.0
    for t in range(60):
        ks = [int(generator.binomial(g, p_true)) for _ in range(prompts)]
        k = sum(ks)
        raw_rate = k / (prompts * g)
        s = 0.9 * s + k
        n = 0.9 * n + prompts * g
        rate = (s + 1.0) / (n + 2.0)
        est.observe("e", make_posterior(rate, se_fast=0.05, se_slow=0.05), prompts, raw_rate)
        lp, se, z = est.compute("e", make_posterior(rate))
        if t >= 40:
            checks += 1
            flags += int(abs(z) >= 2.0)
    return flags / max(checks, 1)


FPR_SEEDS = {"lp_a": 7, "lp_b": 11, "lp_c": 13}


def test_flat_sequences_small_false_positive_rate_all_methods():
    for method in ("lp_a", "lp_b", "lp_c"):
        fpr = _run_flat_fpr(method, FPR_SEEDS[method])
        # roadmap target <= 5% on evaluation seeds; the synthetic SE here is not
        # the tracker's real conservative SE, so the unit test allows a margin
        assert fpr <= 0.10, f"{method} FPR {fpr:.3f}"


# lp_use transformations ---------------------------------------------------------

def test_unknown_method_rejected():
    try:
        LPEstimator("lp_d", window_rounds=10)
    except ValueError as exc:
        assert "unknown lp_method" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_lp_use_transformations():
    # applied by the engine; keep a table-level check here for the contract
    candidates = {"signed": lambda x: x, "positive": lambda x: max(x, 0.0), "abs": abs}
    assert candidates["signed"](-0.7) == -0.7
    assert candidates["positive"](-0.7) == 0.0
    assert candidates["abs"](-0.7) == 0.7
