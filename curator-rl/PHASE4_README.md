# PHASE 4 README — The Signal Engine: Pass Rate, LP, Richness, Proxy Reward, S1–S5 Statuses

This document describes **everything Phase 4 of the CURATOR project built**,
how it works, and why. Phase 3 gave us the algorithm testbed (simulator,
oracles, scenarios, harness). Phase 4 builds the **scheduler's eyes**: the
Signal Engine that converts round observations into the per-environment
signals and statuses the D-UCB scheduler (Phase 5) will consume.

> Formal contract: `docs/SPEC.md` (§5–7), Roadmap Part E (E.2–E.5) and Part P
> Phase 4. All deviations: `docs/DECISIONS.md` D-30 … D-43. Companion
> write-up: `docs/phase_4_explanation.md`.

---

## 1. Where this phase sits

- **Input:** `RoundObservation` streams — the exact B.3 schema the
  simulator (Phase 3) and the real trainer (Phase 8) both emit. The engine is
  L1-pure (imports only `core`, numpy, scipy; the import-rule test enforces
  it) and contains no RNG, so everything is deterministic and replayable.
- **Output:** per-environment `SignalVector` each round (Phase 5's
  `MixtureDecision` and the ROI engine both read it), and the S1–S5 status
  machine that drives exploration quotas, saturation caps, shrinkage and
  calibration requests.
- **Golden rule this phase implements:** every scientific number comes from a
  formula the ROADMAP freeze supplies AND a study measured on the simulator —
  no thresholds were hand-picked this phase.

## 2. What we built, file by file

| File | What it is |
|---|---|
| `src/curator_rl/signals/passrate.py` | `PassRateTracker` + `Posterior` (E.2): three discounted pooled count windows (λ=0.9 main, λ_f=0.7 fast, λ_s=0.95 slow), Beta(1,1) posterior, 68% CI = ±1 posterior σ. | 
| `src/curator_rl/signals/progress.py` | `LPEstimator` (E.3): LP-A (fast − slow; conservative overlapping SE), LP-B (two-stage WLS slope + residual-variance inflation), LP-C (non-overlapping windows on RAW per-round rates). |
| `src/curator_rl/signals/richness.py` | `RichnessEstimator` (E.3.3): mixed / band / variance modes; per-group exact path via `update_groups`. |
| `src/curator_rl/signals/proxy.py` | `ProxyReward` (E.4 / S-5): σ_LP scaling, LP' clip, x = α·LP' + β·SR, cost normalisation with `scheduler.cost_exponent`, [0,1] quantile map (warm-up priors → reservoir percentiles). |
| `src/curator_rl/signals/status.py` | `StatusClassifier` (E.5 / S-1 / S-3): S1–S5 state machine, dwell, hysteresis, S1 re-entry floor, S5 mismatch streaks, `flip_rate()`. |
| `src/curator_rl/signals/engine.py` | `SignalEngine` facade: `update(obs) -> {env: SignalVector}`; calibration bookkeeping + mismatch API; JSON-safe checkpoint state. |
| `src/curator_rl/core/types.py` | `EnvStatus` (StrEnum S1–S5), `SignalVector` (frozen dataclass). `MixtureDecision` stays deferred (D-30). |
| `src/curator_rl/core/config.py` + `configs/base.yaml` | New keys: `signals.window_rounds` (W for LP-B/C), `signals.status.s1_entry_ratio`, `proxy.quantile_prior_lo/hi`. Tuned status values (D-42). |
| `experiments/analysis/common.py` | Study layer (D-38): `RecordingScheduler` episode capture, signal precompute, truth labels, macro-F1 + flip-rate metrics. |
| `experiments/analysis/lp_estimator_study.py` | The E.3 LP selection protocol → `reports/analysis/lp_estimator_study.{json,md}`. |
| `experiments/analysis/tune_status_thresholds.py` | The E.5 threshold-tuning protocol → `reports/analysis/tune_status_thresholds.{json,md}`. |
| `tests/unit/{test_pass_rate,test_progress,test_richness,test_proxy,test_status,test_signal_engine}.py` | 52 new unit tests incl. the roadmap's anti-oscillation test and the repo's first Hypothesis property test. |
| `tests/integration/test_signal_engine_scenarios.py` | Engine over every scenario's real episodes: legal statuses, bounds hold, noisy arm never S3/S4, mid-stream checkpoint forks byte-identical decisions. |

## 3. Key mechanics (the non-obvious parts)

**Pass rate.** Counts decay even on zero-prompt rounds (time passes), but
`rounds_seen` only counts rounds the env actually received prompts — the
status machine later distinguishes "never pushed" from "starved".

**LP — the study caught two real bugs before they reached the paper (D-37):**
LP-B's first SE divided the residual spread by Σw·(k−2) instead of (k−2) —
the SE collapsed by the prompt factor and flat sequences were flagged
"progressing" 100% of the time. LP-C's z assumed independent windows, but
pooled EW windows overlap mechanically → measured FPR 0.40; LP-C therefore
runs on the raw per-round k/n.

**Richness fidelity (D-32).** The aggregate only carries `n_groups_mixed`; for
G = 8 the roadmap's 0.1–0.9 band collapses to `mixed` (S-8), so the aggregate
path is exact for the MVP. Band/variance modes and the exact per-group path
are implemented and unit-tested for later use.

**Proxy reward.** First update has σ=0 by construction → LP'=0 (no overflow).
Unknown unit cost (env never charged) → proxy reward 0, not infinity (D-33);
the warm-up guarantees every env gets allocated before it matters.

**Status machine (E.5, D-36/D-42).** Priority S1 > S5 > S3 > S4 > S2. Entering
S3/S4 needs the condition for h consecutive rounds; leaving needs the
hysteresis leave-rule AND `dwell_min` completed rounds; S2 exits ungated; S1
exit is exempt; S5 enters instantly on the mismatch streak (q windows) and
needs q' clean windows to leave. S1 re-entry needs evidence halved
(`s1_entry_ratio`) — this one change took the 8-env random-mixture flip rate
from 8.1 to ~0 pooled. A forgetting spike cannot re-enter S3 (the S3 exit
falls through to S2/S4, never back to S3).

**S5 is built but unwired (D-34):** the mismatch flag is an API
(`SignalEngine.set_calibration_mismatch`); its data source is the Phase 9
α/β fitter. The state machine and streak gates are fully unit-tested now.

## 4. Study evidence (the Phase 4 exit criteria)

Both studies run on S-A/S-B/S-E/S-G episodes recorded in-process
(tuning seeds 0–9/0–19; evaluation seeds 100–119/100–129 — per Roadmap E.7
the sets are disjoint and both phases are reported).

**LP estimator selection** (`report: lp_estimator_study`):

| estimator | ρ vs true velocity | sign acc | FPR | best lag |
|---|---|---|---|---|
| LP-A | 0.23–0.27 | 0.80–0.82 | **0.104** (fails ≤5%) | 3 |
| **LP-B (chosen)** | 0.19–0.20 | 0.70–0.72 | **0.000** | 3 |
| LP-C | 0.13–0.16 | 0.31 | 0.077 | 1 |

The pre-registered rule picked **LP-B**; the roadmap predicted LP-A, and the
rule was followed as frozen (D-40 documents the FPR miss explicitly instead
of silently overriding).

**Status threshold tuning** (`report: tune_status_thresholds`, 243-point
grid, truth anchored to frozen pre-tuning constants, D-39/D-43):

| phase | picked | macro-F1 | pooled flip /100 rounds |
|---|---|---|---|
| tuning (seeds 0–9) | p_sat 0.7, p_hard 0.05, z_up 1.5, dwell 3, h 4 | **0.9705** | 0.37 |
| evaluation (seeds 100–119) | (same, h argmax 2-vs-4 tie) | **0.9672** | 0.38 |

Exit criteria: macro-F1 ≥ 0.85 ✓, flip ≤ 5/100 ✓. `configs/base.yaml` carries
the tuned values (D-42), with every prior default change traceable to a
committed report rather than taste.

## 5. Documented deviations

D-30 (types), D-31 (richness EW), D-32 (band approximation), D-33 (interim
unit-cost EMA), D-34 (S5 API), D-35 (proxy discount), D-36 (dwell semantics),
D-37 (LP BUGS: SE df + raw-window LP-C), D-38 (in-process study collection),
D-39 (truth-label mirror), D-40 (LP choice), D-41 (pooled flip estimator),
D-42 (tuned thresholds + S1 entry floor), D-43 (frozen truth anchor).

## 6. How to run everything

```bash
python -m pytest                 # 183 tests (unit + contract + architecture + integration)
python -m ruff check src tests experiments scripts

# studies (figures off; JSON + markdown reports only)
python experiments/analysis/lp_estimator_study.py --scenarios S-A,S-B,S-E --tuning-seeds 20 --eval-seeds 30
python experiments/analysis/tune_status_thresholds.py --scenarios S-A,S-B,S-E,S-G --tuning-seeds 10 --eval-seeds 20
```

Reproducibility: episodes are pure functions of (scenario, seed); the study
reports are regenerated exactly by the commands above; engines and studies
are RNG-free.

## 7. What Phase 5 is

Discounted-UCB scheduler (Curator v0) + the Uniform/Static/LP/UCB baselines:
the `MixtureDecision` type, the mixture map with exploration floors + status
constraints, and the Gate 2 simulator validation. Everything Phase 5 needs —
`solve-the-bandit-on-a-simulated-world` — now exists: real signals, real
statuses, and a simulator whose truth is known and replayable.
