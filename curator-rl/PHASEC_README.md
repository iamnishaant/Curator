# PHASE C README — SOTA-style baselines, fair tuning, long-horizon scenarios, Gate 2''

This document describes **everything Roadmap v3 Phase C built**, what it found,
and why. Phase 5 delivered the scheduler and passed Gate 2. Phase C asks the
harder question the literature forces: does Curator beat the *current*
learnability-bandit methods (SEC, DUMP) when every method is tuned fairly, in a
world with real cost spread and a realistic horizon?

> Contract: `docs/ROADMAP_v3.md` §4.4, §6.3 (Phase C), `docs/GATES.md`.
> Decisions: `docs/DECISIONS.md` D-68 … D-71. Evidence: `reports/sweeps/tune_all.md`,
> `reports/gate2/gate2b.md`, `reports/analysis/curator_diagnosis.md`.

---

## 1. Headline result

**Gate 2'' fails, in one specific place, and the cause is known.**

| Criterion (evaluation seeds 100–149) | Result |
|---|---|
| (i) Curator beats SEC-style and DUMP-style in **S-C** | pass: +0.061 [+0.050, +0.072] and +0.059 [+0.048, +0.070] |
| (i) Curator beats SEC-style and DUMP-style in **S-I** | **fail**: −0.008 [−0.013, −0.004] and −0.008 [−0.013, −0.003] |
| (ii) S-J (equal costs): no underperformance vs Uniform | pass: +0.010 [+0.005, +0.015] |
| (iii) original Gate 2 (a)–(e) still hold | pass |
| (iv) junk-arm share | pass (50/50 seeds) |

The roadmap's stop rule applies ("if (i) fails, diagnose before GPU integration").
It was diagnosed (§4); the criterion was **not** relaxed. It becomes the
acceptance test for Phase D: the calibrated Curator must pass (i) on S-I with no
privileged information.

## 2. What was built, file by file

| File | What it is |
|---|---|
| `core/advantage.py` | `group_mean_abs_advantage(k, G)`: mean \|GRPO advantage\| of one binary-reward group, `2p(1-p)/(std+eps)` with TRL's unbiased std; zero for all-pass/all-fail groups |
| `core/types.py` | `EnvRoundObs.sum_abs_adv` (optional trailing field) |
| `simulator/world.py` | emits `sum_abs_adv` per round |
| `scheduler/baselines/sec.py` | `SECStyleBandit`: TD(0) value on mean \|A\| per environment, Boltzmann action, shared floor and warm-up |
| `scheduler/baselines/dump.py` | `DUMPStyleUCB`: UCB on mean \|A\| with undiscounted prompt counts, softmax action |
| `core/config.py`, `configs/base.yaml`, `docs/SPEC.md` | validated `baselines:` section with each method's own hyperparameters (before, UCB and LP silently reused Curator's) |
| `experiments/run_sim.py` | `make_scheduler(..., cfg=)` builds every method incl. `sec`, `dump`, with per-method overrides |
| `simulator/harness.py` | `weight_concentration`: mean max-weight, normalised entropy, share of rounds above 0.5 (D-67) |
| `simulator/scenarios.py`, `configs/sim/scenario_si.yaml`, `scenario_sj.yaml` | S-I (heterogeneous cost, ~70 rounds) and its homogeneous twin S-J |
| `experiments/sweeps/tune_all.py` | fair tuning: 8 pre-registered configs per adaptive method, one-SE + least-concentrated selection |
| `experiments/gate2b_check.py` | the pre-registered Gate 2'' verification |
| `experiments/analysis/curator_diagnosis.py` | component ablation and the privileged-S5 upper bound |
| `tests/unit/{test_advantage,test_sec_dump,test_tuning}.py` | hand-computed expectations for the new code (248 tests in total) |

## 3. Design decisions worth knowing

- **Signal without trainer internals.** For binary rewards the GRPO advantage
  statistic is a closed form of the pass count, so SEC/DUMP-style methods need
  nothing from the trainer. The real reward wrapper will call the same function.
- **Equal effort, different knobs.** Every adaptive method gets 8 configurations
  on the same 50 tuning seeds and the same four tuning scenarios. S-J is held out
  so it is out-of-sample evidence for the "tie when costs are equal" hypothesis.
- **The selection rule was fixed before the sweep:** best objective, then anything
  within one standard error, then the least concentrated mixture. It avoids
  choosing a near-one-hot allocation for a statistically tied score (D-67).
- **S-I and S-J are placeholders** for costs and pass rates until the Phase B
  pilot measures real ones (D-70). They are cost-heterogeneous by construction;
  they are a stress test, not a prediction.
- **A unit bug was found and fixed** in Gate 2(b): the tolerance was 1.0 in score
  units (100 points). It is now 0.01. The criterion still passes (D-69).

## 4. Diagnosis of the S-I failure (tuning seeds, `curator_diagnosis`)

| Finding | Evidence |
|---|---|
| The noisy arm takes 20–25% of Curator's compute before calibration | cheap, rich in signal, indistinguishable from a plateau (E.5, H3) |
| A privileged S5 flag on the noisy arm lifts S-I from +0.011 to +0.030/+0.036; S-J +0.034 → +0.078/+0.084; S-A +0.121 → +0.158/+0.167 | upper bound, not a method |
| Cost normalisation costs 2.9 points in S-I but wins S-C by 13 points | valuable arms (MATH, MBPP) are the expensive ones in S-I; in S-C cost is unrelated to value |
| `hard` statuses cost 1.3 (S-I) and 2.9 (S-J) points vs `soft`, gain 1.3 in S-A | Gate 2(c) needs the S4 cap |
| The best static mixture in S-I is 48% MATH / 27% GSM8K / 25% Countdown / 0% MBPP | it exploits GSM8K↔MATH transfer that LP and richness cannot see |
| Tried and rejected: capping only S4 (`hard_s4`) | did not recover S-I (D-71) |

Reading: Curator's cost lever works where cost varies independently of value, and
the weak spot is exactly the one the project thesis already names: a cheap proxy
needs held-out calibration. Phase D is the fix, and S-I is its test.

## 5. Other observations

- **Concentration (D-67):** UCB and Curator put most of the mixture on one arm in
  S-A and S-C (mean max-weight 0.84–0.89). SEC/DUMP-style stay close to uniform.
  The simulator cannot penalise this; real GRPO could.
- **Static is a fair baseline:** the declared sizes put it near the top of its
  range in S-A (D-63).
- **Tuned values changed:** Curator γ 0.90 → 0.95, κ 0.25 → 0.5 (a statistical
  tie with the old point; the rule preferred the less concentrated one). Gate 2
  was re-run and still passes; the S-A oracle-gap closure is 43%.

## 6. How to run everything

```bash
python -m pytest                                   # 248 tests
python -m ruff check src tests experiments scripts
python experiments/sweeps/tune_all.py --seeds 50   # fair tuning, ~25 s on 16 cores
python experiments/gate2b_check.py --seeds 50      # Gate 2'' (also re-runs Gate 2)
python experiments/analysis/curator_diagnosis.py --seeds 40
```

## 7. What comes next

Phase D (calibration, ROI and LOO on the simulator): C1/C2/C2d credit, the α/β
fitter, the S5 detector wired to `SignalEngine.set_calibration_mismatch`, and the
power study. Acceptance: Gate 6-sim and **Gate 2'' (i) on S-I**. Phase B (TRL
spike, cost meter, Kaggle pilot) supplies the real costs and pass rates that will
replace S-I's placeholders.
