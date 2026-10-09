# Phase C explanation — SOTA-style baselines and fair tuning (Roadmap v3 Phase C)

Short companion to `PHASEC_README.md`. Decisions: `docs/DECISIONS.md` D-68 … D-71.

## One-sentence summary

Phase C adds the SEC-style and DUMP-style learnability bandits as baselines,
tunes every adaptive method with the same budget and a pre-registered selection
rule, adds a long-horizon cost-heterogeneous scenario (S-I) and a homogeneous twin
(S-J), and runs Gate 2'': Curator beats the learnability bandits where cost varies
independently of value (S-C) but not in S-I, and the diagnosis points at
uncalibrated spurious-reward detection.

## Module map

```
core/advantage.py              mean |GRPO advantage| of a binary group (closed form)
scheduler/baselines/sec.py     TD(0) on mean |A|, Boltzmann
scheduler/baselines/dump.py    UCB on mean |A|, softmax
core/config.py / base.yaml     baselines.{ucb,lp,sec,dump}: per-method hyperparameters
simulator/harness.py           weight_concentration (D-67)
configs/sim/scenario_si|sj     heterogeneous-cost portfolio and its equal-cost twin
experiments/sweeps/tune_all.py fair tuning, one-SE + least-concentrated selection
experiments/gate2b_check.py    Gate 2''
experiments/analysis/curator_diagnosis.py   ablation + privileged-S5 upper bound
```

## Results at a glance

- Curator vs SEC/DUMP-style: +0.06 in S-C (pass), −0.008 in S-I (fail).
- Homogeneous twin S-J: Curator is above Uniform by +0.010 (expected tie or better).
- Original Gate 2 still passes with the re-tuned values (γ 0.95, κ 0.5).
- Perfect spurious-reward detection would lift Curator to the SEC/DUMP level in S-I
  (+0.030–0.036 vs +0.020–0.031 on tuning seeds), but not to Standard UCB (+0.053).

## Honest caveats

- S-I and S-J use **placeholder** costs and pass rates until the Phase B pilot.
- SEC and DUMP are **re-implementations from the papers' descriptions**, not the
  authors' code, so "beats SEC" means "beats an SEC-style bandit".
- The upper-bound experiment uses privileged information and is not a method.
- Standard UCB is the strongest baseline in S-I and S-J; the cost-aware claim has
  to be carried by cost-heterogeneous portfolios and by calibration.
- Allocation concentration is high for UCB and Curator (D-67).

## What Phase C does not touch

Calibration, the α/β fit, the ROI engine, the cost meter and every trainer file.
