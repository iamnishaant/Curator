# EXPERIMENT PROTOCOL

Source: Roadmap v2.0 Parts L, H.6, Q; v3 §3, §4.8, §8. This file is the pre-registration surface: values here are fixed **before** the runs they govern. Changes after a freeze need a dated entry in `docs/DECISIONS.md` naming the claims affected.

## 1. Matched compute

- Every method runs to the same budget B under the ledger rule (SPEC §4): stop at the first optimizer step where cumulative charged cost ≥ B; overshoot is logged and at most one step.
- Charged: training, scheduler-owned calibration evaluations, Curator overhead. Not charged (identical for every method): reporting evaluations.
- `stop_on_target` is OFF in every matched comparison.
- Headline plots are score versus GPU-dollar; steps, rollouts and tokens are reported as covariates.

## 2. Fairness table

| Item | Rule |
|---|---|
| Base model, initial checkpoint, LoRA init seed | identical across methods |
| Environment pool, calibration/dev/test sets, decoding | identical |
| Optimiser | identical; constant learning rate after warmup; no schedule depends on step count |
| Seeds | same list for every method (paired); chosen before the first run |
| Tuning | equal budget per adaptive method, on simulator tuning seeds (0–49) and the `dev` split only |
| GPU type | all runs in one comparison use the same GPU type (Kaggle T4) |
| Unavoidable differences | step and token totals (reported); calibration cost exists only for methods that calibrate (charged) |

## 3. Equivalence margins (Gate 4 and H6)

Equivalence is claimed only when the paired difference lies inside a margin registered here; a non-significant test is never evidence of equivalence. Distribution tests (KS) are diagnostics only.

| Metric | Margin δ | Used by |
|---|---|---|
| Final benchmark score (macro, benchmark points) | δ_S = 1.0 | Gate 4 passthrough equivalence; H6 (homogeneous regime) |
| Mean per-step training reward | δ_R = 0.01 | Gate 4 |
| Realised environment share vs requested weight | ±2 percentage points over 100 rounds | Gate 4 |
| Weight-update lag | ≤ 1 optimizer step | Gate 4 |
| Curator overhead | < 2% of round time | Gate 4 |

## 4. Statistics

- Paired differences per seed; 95% percentile bootstrap, 10,000 draws; seeds are the resampling unit.
- Co-primary hypotheses H1 and H2 (v3 §3.3) use a Holm correction. Everything else is secondary.
- No claim of improvement when intervals overlap. Mean ± std and interval are reported for every cell.
- Simulator claims use ≥ 50 evaluation seeds starting at 100 (tuning seeds are 0–49). Real runs: 5 paired seeds for the main matrix, 3 for ablations, regime H6 and LOO.
- ROI is described as "estimated marginal contribution under the mixtures actually run", never as causal.

## 5. Primary metric (to be frozen at Gate 10 for Level 1)

Macro-average of the domain test-set scores at budget B (domains equally weighted; `noisy` and `too_hard` have no test set). Secondary: compute-to-target, AUC of score versus GPU-dollar, per-domain scores. Every run reports every domain; no post-hoc subset summaries.

## 6. Run hygiene

- A clean Git tree and a frozen config hash are required for every GPU matrix (Gate 10); the hash and commit are recorded in `metadata.json`.
- Crashed runs are re-run with the same seed. Diverged runs (KL blow-up, NaN) are reported, not dropped. The number of runs is fixed in advance; no seed is chosen after seeing results.
- The sealed test set is opened once per final checkpoint through `evaluation.guard`.

## 7. Simulator evaluation discipline

- Tuning seeds 0–49; evaluation seeds 100–149 (disjoint).
- Truth anchors for status labels are frozen constants inside the study scripts (D-43), not read from `base.yaml`.
- Gate criteria are committed before the gate is run; a result near a threshold is reported with its interval and handled by the D-18 procedure, not by moving the threshold.
