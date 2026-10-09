# Phase 5 explanation — the scheduler (v1 Phase 5 / v2 Phases 4–6)

Short companion to `PHASE5_README.md`. Formal contract: `docs/SPEC.md`,
Roadmap Part E/J. Decisions: `docs/DECISIONS.md` D-44 … D-51, D-63 … D-67.

## One-sentence summary

Phase 5 turns the Signal Engine's per-environment signals into a mixture over
environments every round (discounted UCB → z-scored softmax → exploration floor
→ status constraints), ships the Uniform / Static / LP / Standard-UCB baselines,
tunes the hyperparameters on tuning seeds, and passes Gate 2 on 50 disjoint
evaluation seeds.

## Module map

```
scheduler/mixture.py   softmax_floor + apply_status_constraints  (pure functions)
scheduler/ducb.py      DiscountedUCB: Ñ_i, Σ̃_i, scores, MixtureDecision, checkpoints
scheduler/curator.py   Curator facade (v0: proxy + cost + D-UCB + statuses)
scheduler/baselines/   static, lp, ucb (+ uniform, random from Phase 3)
core/quotas.py         largest-remainder quotas (shared by scheduler and trainer)
experiments/gate2_check.py     pre-registered Gate 2 verification
experiments/sweeps/tune_ducb_sim.py   gamma/kappa/tau/epsilon grid on tuning seeds
```

## Results at a glance

> **Note (Phase C):** the numbers below are for the Phase-5 configuration (γ 0.90, κ 0.25). The Phase C re-tune (D-68) moved Curator to γ 0.95, κ 0.5; the current Gate 2 numbers are in `docs/GATES.md` (all five criteria still pass; S-A oracle-gap closure 43%).

Gate 2 criteria (a)–(e) all pass at 50 evaluation seeds. Curator beats Uniform
by +0.066 in S-A, +0.014 in S-B and reaches the S-C target 25.6% sooner. It
leads Standard UCB in S-A and S-C and trails it by 0.008 in S-B. The oracle-gap
development target (75%) is missed at 46–48%; the 25-round horizon caps even a
perfect post-warm-up scheduler at 75.9%, so about half of the gap is
structural (D-65).

## Honest caveats

- **Near one-hot allocation (D-67).** Under the tuned τ=0.3, most rounds put
  more than half the mixture on one arm. The simulator cannot punish this.
- **Standard UCB beats Curator in S-B** where costs are nearly equal; the
  cost-aware claim is carried by S-C. Real portfolios must have cost spread
  (Roadmap v3 §4.1).
- **Static depends on the declared sizes (D-63).** A permutation check puts the
  declared assignment near its best case.
- **Tuning used `soft`; the default is now `hard` (D-64).** Phase C re-tunes
  with `status_control` in the grid.
- **The Gate 2 junk criterion initially failed** on stale code; the history is
  in D-64 and D-66 so it cannot be mistaken for a result.

## What Phase 5 does not touch

Calibration (α/β fit, S5 data source), the ROI engine, the cost meter and any
trainer code. `Curator.update_calibration` stores the observation only; S5 stays
an API until Phase D.
