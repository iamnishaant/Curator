# PHASE 5 README — The Scheduler: Discounted UCB, Mixture Map, Baselines, Gate 2

This document describes **everything Phase 5 of CURATOR built and closed out**,
how it works, and why. Phase 3 gave the testbed (simulator, oracles), Phase 4
the scheduler's eyes (Signal Engine). Phase 5 is the **decision-maker**: it
turns signals into a mixture over environments each round, ships the four
comparison baselines, and passes Gate 2 on the simulator before any GPU is used.

> Contract: `docs/SPEC.md`, Roadmap Part E.5–E.7 and Part J. Deviations and
> decisions: `docs/DECISIONS.md` D-44 … D-51 (built), D-63 … D-67 (Phase A
> close-out). Evidence: `docs/GATES.md`, `reports/gate2/`, `reports/sweeps/`.

---

## 1. Where this phase sits

- **Input:** `RoundObservation` (the schema shared by the simulator and, later,
  the real trainer). **Output:** a weight dictionary per round plus a
  `MixtureDecision` record (UCB components, statuses, intents, prompt quotas).
- The scheduler is L1-pure: no torch, no RNG (Curator), no hidden simulator
  state. It owns its own `SignalEngine`, which required relaxing the import
  rule from "L1 imports only `core`" to "L1 may import `core` and other L1
  packages" (D-48).

## 2. What exists, file by file

| File | What it is |
|---|---|
| `scheduler/ducb.py` | `DiscountedUCB`: discounted pull mass `Ñ_i` and reward mass `Σ̃_i` from the realised exposure and the proxy reward r̄; score `μ̂ + κ·sqrt(ln max(Ñ, e)/Ñ_i)`; warm-up (uniform for `warmup_rounds`); status constraints; JSON-safe `get_state`/`load_checkpoint` |
| `scheduler/mixture.py` | Pure functions. `softmax_floor`: score_norm (none / zscore / rank) → clip(z/τ, ±50) → log-sum-exp softmax → floor ε/N. `apply_status_constraints`: S3/S4/S5 multipliers (soft) or caps (hard), S1 exploration quotas with a mass budget, proportional redistribution, floors re-imposed, sum = 1 within 1e-12 |
| `scheduler/curator.py` | `Curator` facade (Curator v0: proxy + cost + discounted UCB + statuses); `get_roi` is a stub until the ROI engine |
| `scheduler/baselines/static.py` | Static mixture: fixed, pre-registered weights |
| `scheduler/baselines/lp.py` | Learning-progress curriculum (Graves/TSCL): Boltzmann over `|LP|`, cost-blind, same floor |
| `scheduler/baselines/ucb.py` | Standard UCB: Curator with `gamma=1`, `cost_exponent=0`, `status_control=off` |
| `core/quotas.py` | Largest-remainder prompt quotas (moved from the simulator so L1 can use it) |
| `core/types.py` | `MixtureDecision` (ends the D-26 deferral) |
| `core/config.py`, `configs/base.yaml` | New keys `s3_multiplier`, `s4_multiplier`, `s5_shrink`, `s1_quota`; tuned `gamma 0.90`, `exploration_coef 0.25`, `tau 0.3`; `status_control: hard` |
| `simulator/harness.py`, `scenarios.py` | Uncharged reporting evaluations (`report_eval`, D-46) and the `nominal_size` field (D-63) |
| `experiments/run_sim.py` | `make_scheduler` builds every method from `base.yaml` overrides (D-44); `_static_weights` (D-63) |
| `experiments/sweeps/tune_ducb_sim.py` | The γ/κ/τ/ε tuning grid on tuning seeds (D-49) |
| `experiments/gate2_check.py` | The pre-registered Gate 2 verification on evaluation seeds |
| `tests/unit/{test_ducb,test_mixture}.py`, `tests/integration/test_schedulers_sim.py` | UCB hand-computed scores, mixture-map properties, end-to-end runs |

## 3. Key mechanics (the non-obvious parts)

- **Warm-up then UCB.** The first `warmup_rounds` (6) are uniform so costs and
  signals initialise. Afterwards z-scored UCB scores go through a temperature
  softmax. Unexplored arms are never given `score = ∞`; the S1 exploration
  quota handles them (E.6).
- **Statuses act after the softmax.** S3/S4/S5 arms get a multiplier (soft) or
  a cap (hard: S3 at 2× floor, S4 at the floor); released weight is
  redistributed to uncapped arms; floors and S1 quotas are re-imposed; the sum
  stays 1.
- **Anti-flap and starvation rules (D-51).** An established S3/S4 arm is not
  demoted to S1 by evidence decay, otherwise a starved too-hard arm cycled
  S1 → 30–60% quota → S4 → starved → S1. S1 keeps priority over S5 for fresh
  arms; S5 still overrides a saturated incumbent.
- **Static baseline must differ from Uniform (D-63).** It was identical to
  Uniform on S-A/S-B/S-C because no benchmark weights were set. It is now
  size-proportional on declared dataset sizes. A sensitivity check over 20
  random size assignments showed the declared assignment is near Static's best
  case, so the comparison is conservative for Curator.

## 4. Gate 2 — evidence (evaluation seeds 100–149)

> **Note (Phase C):** the numbers below are for the Phase-5 configuration (γ 0.90, κ 0.25). The Phase C re-tune (D-68) moved Curator to γ 0.95, κ 0.5; the current Gate 2 numbers are in `docs/GATES.md` (all five criteria still pass; S-A oracle-gap closure 43%).

| Criterion | Result |
|---|---|
| (a) S-A, Curator − Uniform | +0.0661, 95% CI [0.0531, 0.0785] — pass |
| (b) S-B, Curator − Uniform | +0.0137, CI [0.0093, 0.0182]; lower bound > −1 point — pass |
| (c) too-hard arm share (last 40% of rounds) | mean 0.0144; 48/50 seeds ≤ 1.5 × floor (≥ 90% required, 0.0188 threshold) — pass |
| (d) status macro-F1 / flip rate | 0.932 / 0.0 per 100 rounds — pass |
| (e) S-C compute-to-target vs Uniform | +25.6%, CI [21.5%, 29.4%] — pass |
| dev target: oracle-gap closure, S-A | 46–48% against a 75% target — **missed**, analysed (D-65) |

| Final score | Uniform | Static | LP | Std UCB | Curator |
|---|---|---|---|---|---|
| S-A | 0.579 | 0.608 | 0.597 | 0.611 | **0.645** |
| S-B | 0.405 | 0.418 | 0.418 | **0.427** | 0.419 |
| S-C | 0.344 | 0.347 | 0.351 | 0.359 | **0.400** |

**Why the oracle gap is only 48% (D-65).** An S-A episode is 25 rounds and the
first 6 are the uniform warm-up. A scheduler that plays Uniform for 6 rounds
and the privileged best-static mixture afterwards closes at most 75.9% of the
gap (64.8% with 9 uniform rounds, 49.1% with 12). Curator reaches 63% of its
own ceiling. The horizon is a scenario property; Roadmap v3 §4.2 requires at
least 60 rounds in real runs and Phase C re-parameterises the scenarios.

## 5. What the close-out found (Phase A, 2026-10-09)

| Finding | Action |
|---|---|
| 4 failing tests (stale `gamma` asserts, S1/S5 priority) | tests updated; priority restored to the roadmap's order; 4 new regression tests |
| Gate 2 report on file said FAIL for the junk share | it was stale (pre-D-51 code). At 50 seeds `soft` failed (38/50), `hard` passes (48/50): D-64, chosen on tuning seeds and confirmed on evaluation seeds |
| Suspected "S1 re-entry pump" | traced and measured: ≤ 0.72% of the budget; **left unchanged** (D-66) |
| Allocation is nearly one-hot | measured (97% / 61% / 96% of rounds have a max weight above 0.5); logged as D-67 and sent to Phase C tuning |
| Static baseline equal to Uniform | fixed (D-63) |
| D-44…D-51 cited in code but unlogged | written |
| `scratch_debug5/6/7.py`, 8 lint errors | removed / fixed |

## 6. How to run everything

```bash
python -m pytest                       # 217 tests
python -m ruff check src tests experiments scripts
python experiments/sweeps/tune_ducb_sim.py --seeds 10          # D-UCB tuning (tuning seeds)
python experiments/gate2_check.py --seeds 50                   # Gate 2 (evaluation seeds 100-149), ~15 s
python experiments/run_sim.py --scenario configs/sim/scenario_sa.yaml \
    --methods uniform,static,lp,ucb,curator --seeds 20
```

## 7. What comes next

Roadmap v3 Phase B (TRL spike, cost meter, pilot on Kaggle) and Phase C
(SEC/DUMP-style baselines, equal tuning for every adaptive method, a
concentration constraint, scenarios re-parameterised to the measured horizon,
Gate 2''). See `docs/ROADMAP_v3.md`.
