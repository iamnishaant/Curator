# Phase 4 explanation — the Signal Engine (v1 Phase 4 / v2 Phase 5-signals)

Short companion to `PHASE4_README.md`; the full behavioural contract is
`docs/SPEC.md` + the Roadmap Part E, and every documented decision this phase
took is `docs/DECISIONS.md` D-30 … D-43.

## The one-sentence summary

Phase 4 turns each `RoundObservation` (the schema shared by the simulator and
the future real trainer) into a per-environment **SignalVector** every round —
pass rate, learning progress, signal richness, proxy reward, unit cost, and a
status S1–S5 — with the LP estimator and the status thresholds **chosen by
pre-registered studies on the simulator**, not hand-picked.

## Module map

```
signals/passrate.py   PassRateTracker  — E.2: three discounted pooled count
                      pairs (λ=0.9, λ_f=0.7, λ_s=0.95), Beta(1,1) posterior,
                      group-level SE (the prompt group is the unit; rollouts
                      inside a group are correlated).
signals/progress.py   LPEstimator      — E.3: LP-A fast-slow (conservative
                      overlapping-window SE), LP-B two-stage weighted slope
                      with residual-variance inflation (autocorrelation!),
                      LP-C non-overlapping windows on RAW per-round rates.
signals/richness.py   RichnessEstimator — E.3.3: mixed/band/variance modes;
                      mixed is exact from the round aggregate; band uses the
                      posterior-overlap estimate; variance decides on the
                      discounted group-rate std (D-31/D-32).
signals/proxy.py      ProxyReward      — E.4: σ_LP (discounted), LP' clip/σ
                      (+L=3), x = α·LP' + β·SR, r̃ = x/c̃^θc, r̄ = quantile map
                      (warm-up priors → reservoir percentiles, D-35).
signals/status.py     StatusClassifier — E.5: S1–S5 state machine with
                      priority S1>S5>S3>S4>S2, dwell on S3/S4 exits,
                      p/sr hysteresis, S1 re-entry floor (s1_entry_ratio),
                      S5 mismatch streaks (engine API, wired in Phase 9),
                      and flip_rate().
signals/engine.py     SignalEngine     — the facade: update(RoundObservation)
                      → {env: SignalVector}; update_calibration,
                      set_calibration_mismatch, JSON-safe get_state/
                      load_checkpoint. Deterministic, no RNG, sorted env
                      iteration (permutation equivariance is property-tested).
```

## What Phase 4 does NOT touch

The scheduler (Phase 5 will consume `SignalVector`), the harness/simulator
(studies collect episodes in-process via a recording scheduler, D-38), the
cost meter (the engine's interim unit-cost EMA is D-33), MixtureDecision
(D-30), and the S5 mismatch data source (D-34; exists as an API + streak
state machine only).

## Study evidence (why the numbers are what they are)

- **LP selection** (`experiments/analysis/lp_estimator_study.py`,
  `reports/analysis/lp_estimator_study.{json,md}`): the pre-registered rule —
  best Spearman ρ vs true skill velocity subject to FPR ≤ 5% and lag ≤ 5 —
  selects **LP-B** (ρ 0.19–0.20, FPR 0.000, lag 3, sign accuracy ~0.7).
  LP-A has better ρ (0.23–0.27) but its measured absolute SE overruns the 5%
  FPR constraint (0.104) — kept as fallback only. Two implementation bugs were
  caught by the FPR measurement (D-37) before they could reach the paper.
- **Threshold tuning** (`experiments/analysis/tune_status_thresholds.py`,
  `reports/analysis/tune_status_thresholds.{json,md}`): 243-point grid,
  truth anchored to frozen constants (D-43) with S1 truth mirroring the
  predictor's own discounted evidence band (D-39): winner
  p_sat 0.7 / p_hard 0.05 / z_up 1.5 / dwell 3 / h 4 → F1 0.9705 (tuning
  seeds 0–9), 0.9672 (evaluation seeds 100–119), pooled post-warm-up flip
  rate 0.37–0.38 per 100 rounds. Both Phase 4 exit criteria met
  (macro-F1 ≥ 0.85, flip ≤ 5/100).

## Good-to-know gotchas found in validation

- The "flip ≤ 5" metric is sensitive to two estimator choices: counting the
  designed warm-up S1 handover as a flap, and averaging per-env rates with
  2-interval tails. Both removed (D-41 / D-42's s1_entry_ratio).
- The S1 entry floor means a starved arm does NOT instantly flip back to S1 —
  it needs evidence halved — which massively stabilised the 8-env random
  mixtures.
- LP-C's independent-sample z is only valid on raw per-round rates; pooled
  windows overlap mechanically and produce FPR 0.40 (D-37).
