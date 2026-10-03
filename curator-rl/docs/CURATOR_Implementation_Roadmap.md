# CURATOR: Implementation Roadmap (v2.0)

Cost-Aware Environment Selection for Agentic RL Training of LLMs

**Status:** roadmap only. No project code is written yet. Section "First implementation step" at the end specifies Phase 1 and then stops.

**v2.0 changes.** This revision adds a hard MVP/stretch scope boundary (Part A0), renumbers the phase map around the MVP (Part A.3), defines team ownership and a 12-week calendar (Part A.4), adds a Windows/WSL2 platform plan (Part O), rewrites Gate 2 and Gate 4 on statistically defensible grounds, and adds an Experiment Freeze Gate before the expensive runs. All technical detail from Parts B–U (v1.0) is preserved. **Phase numbers inside Part P and in the defect table are v1.0 numbers; the v2.0 mapping is given in Part A.3 and at the top of Part P.**

**Inputs used:** your submitted proposal (Sections 1–7), your Part 2 (formulation, diagram, dataset plan, metrics, novelty), and your roadmap brief.

**Conventions.** MUST and MUST NOT are hard rules enforced by tests or gates. SHOULD is a default that can be changed with a written reason in `docs/DECISIONS.md`. Numeric defaults are starting values to be set by pilots. They are never hard-coded in Python (Rule 5). Items marked **[verify]** depend on external libraries or datasets whose current state must be checked when the phase starts.

---

## Specification review: defects found before any code is written

The brief asks me to find weak or ambiguous parts of the proposal instead of silently implementing them. The table lists each one with the resolution this roadmap adopts. All of them are frozen into `docs/SPEC.md` in Phase 0. **Phase numbers in this table use the v1.0 numbering; see the v2.0 mapping in Part A.3.**

| ID | Where | Problem | Resolution adopted | Phase |
|---|---|---|---|---|
| S-1 | Part 2, 9.1 | S6 (budget exhausted) and S7 (target reached) are listed as environment states, but they are conditions of the whole run | Two separate types: `EnvStatus` ∈ {S1..S5} per environment, and `RunTerminal` ∈ {T_budget, T_target} per run | 0 |
| S-2 | Part 2, S7 | Early stop on "target achieved" breaks matched-compute comparisons | `stop_on_target` is OFF in every matched comparison. Compute-to-target is computed post hoc from the score-vs-cost curve. The flag is for demos only | 0, 14 |
| S-3 | Part 2, 9.2 | Per-state action menus (A1…G3) are discrete, but the formal action is a continuous mixture w_t | The menus become an interpretation layer. `AllocationIntent` labels are derived from (status, Δw). Statuses also impose soft constraints (exploration quota, floor cap, shrinkage). The only optimised action is w_t | 4, 5 |
| S-4 | Part 2, 9.4 | Symbol clashes: c is both exploration coefficient and cost c_i. γ is a bandit discount (the MDP discount is 1). T is both horizon and current round | Config names: `exploration_coef` (κ in math), `gamma` (documented as bandit forgetting factor), `round_t`, `max_rounds`, `n_eff_total` for the log argument | 0 |
| S-5 | 9.3 | Proxy r̃ = (α·LP + β·SR)/c has arbitrary scale, but a UCB bonus assumes rewards in a known range | Cost is normalised by its running mean across environments. The reward is mapped to [0,1] by a robust running quantile and clipping. Both steps are configurable and unit-tested | 5 |
| S-6 | 9.4 | softmax(UCB/τ) is scale-dependent, so τ means different things on different scales | Scores are z-scored (or rank-transformed) before the softmax. `score_norm: none \| zscore \| rank` is an ablation | 5 |
| S-7 | 9.3 | Learning progress from consecutive batch pass rates is confounded (different prompts, different n) and noisy | Beta-smoothed, discounted pooled pass rates; LP is a fast-minus-slow estimate with a standard error. The estimator is chosen on the simulator, where truth is known | 4 |
| S-8 | 7.1 | The signal-richness band [0.1, 0.9] on a group pass rate is degenerate for small group size G (with G = 8 it means "mixed group") | SR is defined at group level. The band is configurable and reported with G. A variance-based mode handles continuous rewards | 4 |
| S-9 | 9.3 | r_cal = ΔS · share_i / c_i is not a valid attribution (analysis in Part I) | Kept as baseline credit rule C1 and as the early-training fallback. The main calibrated credit is windowed regression C2, with per-domain slices | 9 |
| S-10 | Sections 2 vs 9.5 | The problem statement says "improvement per unit compute"; the objective says "maximise improvement subject to cost ≤ B". These differ: a pure ratio rewards stopping early | The primary objective is fixed-budget: maximise final score at B. Ratio quantities are reported metrics and leaderboard values | 0 |
| S-11 | 9.1 | One "held-out benchmark" is used for the state, for calibration, for ROI, and for reporting | Four splits: `train`, `calib` (drives scheduler, α/β, and S5), `dev` (threshold and hyperparameter tuning, validation curves), `test` (sealed). Test is used once | 2 |
| S-12 | S5 definition | "Proxy disagrees with benchmark" is defined per environment, but ΔS is global | Calibration benchmark has one slice per environment domain. Disagreement is a z-score of proxy-implied gain against the regression estimate (posterior standard error) | 9 |
| S-13 | Budget | "Σ cost(w_t) ≤ B" has no granularity rule and no statement of what is charged | Stop at the first optimizer step where cumulative cost ≥ B. Training, scheduler-owned calibration evals, and Curator overhead are all charged. Reporting evals are uncharged for every method | 6 |
| S-14 | Fairness | A learning-rate schedule tied to total steps makes methods with cheaper steps train differently | Constant LR after warmup. No schedule may depend on planned step count | 7 |
| S-15 | Part 2, refs | Reasoning Gym is attributed to "Kazemian et al." (it is Stojanovski et al.). [15] and [16] are duplicates. [13] is missing. BFCL is not cited | Fix in the document. Not a code issue, but fix it before submission | 0 |

---

# PART A0: Scope definition — MVP and stretch (v2.0)

## A0.1 The hard MVP boundary

The single biggest risk to this project is not methodological weakness — it is scope consuming the semester before the core result exists. Therefore:

> **MVP = the complete, paper-ready project.** Curator scheduler + GRPO training + matched-compute budget ledger + held-out calibration + ROI engine + Tier-1 comparison of four allocation methods (Uniform, Static mixture, Learning-progress, Full Curator) + 3 seeds + Tier-1 leave-one-out validation, ending with one sealed untouched test evaluation and the figures/tables of Part S.

**MVP research question:**

> Can adaptive, cost-aware environment allocation improve benchmark gain per unit compute compared with fixed and learning-progress-based allocation under a matched compute budget?

Everything below the MVP line increases the quality of the paper but is never needed to rescue it. On MVP completion, remaining work is explicitly labeled **stretch**; it is never silently treated as required.

## A0.2 Component scope table

| Component | MVP | Stretch |
|---|---:|---:|
| Simulator with known dynamics | ✓ | |
| Oracles (O2 best static, O3 myopic marginal-ROI; O1 DP optional) | ✓ | DP oracle |
| Uniform baseline | ✓ | |
| Static mixture baseline | ✓ | |
| Learning-progress baseline | ✓ | |
| Curator / discounted UCB | ✓ | |
| Cost measurement + budget ledger | ✓ | |
| GRPO + TRL integration | ✓ | |
| Held-out calibration (C1 + C2) | ✓ | |
| ROI engine + leaderboard | ✓ | |
| Environment statuses (S1–S5) | ✓ | |
| 3 seeds | ✓ | 5 seeds |
| GSM8K | ✓ | |
| Countdown | ✓ | |
| Noisy-reward environment | ✓ | |
| Final untouched test (sealed) | ✓ | |
| Offline figures + HTML leaderboard (viz/) | ✓ | |
| Basic live weight/cost monitor | | ✓ |
| Full live dashboard with alerts | | ✓ |
| LOO validation | | ✓ Tier 1 done in MVP; Tier 2 stretch |
| MATH environment | | ✓ |
| MBPP (sandboxed code) | | ✓ |
| Knights and Knaves | | ✓ |
| Tool-use environment | | ✓ |
| Too-hard environment | | ✓ |
| EXP3 variant | | ✓ |
| Discounted Thompson sampling | | ✓ |
| Sliding-window UCB | | ✓ |
| SEC-style baseline | | ✓ |
| Large ablation grid | | ✓ |
| verl adapter | | ✓ |
| Trace replay (Tier 1R) | | ✓ |
| Tier 2 (1.5B) experiments | | ✓ |

## A0.3 Tier-1 MVP comparison (the result that completes the paper)

| Method | Reward | Discount | Cost norm | Calibration | Statuses |
|---|---|---|---|---|---|
| Uniform | — | — | — | — | — |
| Static mixture | — | — | — | — | — |
| Learning-progress | LP | — | no | no | no |
| **Full Curator** | LP + SR | γ < 1 | yes (θc = 1) | yes (every K) | yes |

Protocol: 3 paired seeds, identical budget B under the ledger rule (H.6), calibration and Curator overhead charged to Curator only, headline = benchmark score vs GPU-dollar, sealed test used once per final checkpoint.

## A0.4 What is dropped in the MVP, and why it is safe to drop

- **MATH / MBPP / K&K / tool-use / too-hard environments.** The story is allocation strategy, not environment diversity. Three environments with different costs and a noisy sanity arm already exercise every scheduler decision path (learn, saturate, noise) — and the simulator scenarios cover the harder dynamics (transfer, interference, drift, revival).
- **EXP3 / DTS / SW-UCB variants.** Non-stationarity value is tested by the discounting ablation (`gamma: 1.0`) and drift scenarios S-E/S-H in the simulator.
- **SEC-style baseline.** It requires trainer-internal advantage statistics and isolates nothing that the LP baseline doesn't already isolate.
- **Large ablation grid.** The MVP keeps the priority ablations that are free by re-analysis (C1 vs C2 on logged windows) and, if the schedule allows, `cost_exponent: 0` and `calib.enabled: false` on Tier 1.
- **Tier 2 (1.5B).** Tier 1 on 0.5B carries the MVP result. Tier 2 strengthens it, it does not make a new claim.
- **verl adapter and full dashboard.** Proposal wording is corrected to match (A0.5).

## A0.5 Proposal alignment (done in Phase 0)

The proposal promises "a live dashboard with live weights, ROI ranking and alerts" and "a plug-in sampler for TRL and verl". Change the proposal text to:

> **Monitoring:** offline experimental dashboard — figures and an HTML leaderboard of allocation weights, compute cost, benchmark improvement, and estimated ROI, regenerated from run logs.
> **Integration:** initial implementation through TRL, with verl support planned as an extension.

Record both edits in `docs/DECISIONS.md` at Gate 0. If the team later reaches stretch Phases 15–16, the wording can be strengthened back.

---

# PART A: Executive implementation strategy

## A.1 What is being built

Curator is a scheduling layer around an unmodified GRPO trainer.

- **Inner system:** GRPO post-training of Qwen2.5-0.5B/1.5B-Instruct with LoRA, using verifiable rewards from N environments.
- **Outer system:** a non-stationary, budget-constrained bandit (resource-allocation) problem. Each round it chooses a mixture over environments from cheap online signals, and every K rounds it corrects those signals against a held-out calibration benchmark.
- **Outputs:** a trained model, a measured cost ledger, and an environment-level ROI leaderboard, validated against leave-one-out (LOO) contributions.

One modelling caveat is carried through the whole roadmap. Arm values depend on the scheduler's own earlier choices, because training on an environment changes the model. Standard bandit regret bounds assume drift that does not react to the learner. So bandit theory only motivates the design. Evidence comes from the simulator and the real runs.

## A.2 Ten strategic decisions

| # | Decision | Why |
|---|---|---|
| 1 | The scheduler is a pure function of observations (`RoundObservation` in, `MixtureDecision` out). It never imports torch, TRL, or a dataset | It can be tested on a simulator, replayed on traces, and plugged into TRL or verl |
| 2 | Simulator before GRPO (Rule 3). No GPU experiment until Gate 2 passes | The simulator knows the true optimum. Real runs do not |
| 3 | Start with 3 environments (GSM8K, Countdown, noisy-reward). Add the other five after Gate 4 | Rule 2. Keeps the first integration debuggable |
| 4 | Every environment reports through one interface, and every cost comes from measured spans | Rules 7 and 11 |
| 5 | Matched compute is enforced by a budget ledger, not by "same number of steps" | Section 24 of the brief |
| 6 | Four dataset splits with hash-based assignment, committed manifests, and a sealed test set guarded in code | Leakage is the most damaging failure |
| 7 | Build Curator in layers: v0 (proxy + D-UCB) → +cost → +calibration → +status logic. Each layer is a config flag | Makes the ablations free and finds bugs early |
| 8 | The TRL-specific code lives in one file (`trainer/trl_adapter.py`) behind a spike and a pinned version | TRL is at v1.x and still changing. Documented hooks: reward functions receive `prompts`, `completions`, `completion_ids` and all dataset columns; `callbacks`; `rollout_func`; `IterableDataset` **[verify against the pinned version]** |
| 9 | Every expensive run is resumable. Scheduler state, RNG streams, ledger, and LoRA/optimizer state are checkpointed atomically | Rule 10 |
| 10 | The analysis plan is committed before the final runs (Gate 10) | Prevents choosing metrics after seeing results |

## A.3 Phase map (v2.0 numbering)

The phases are renumbered around the MVP (A0.1). The "v1 Phase" column maps to the detailed sections in Part P, so no detail is lost. Owner column: P1 = Algorithm, P2 = Training, P3 = Evaluation/Research (A.4).

| v2 Phase | Name | v1 Phase(s) | Rough person-days | Needs GPU | Owner |
|---|---|---|---|---|---|
| 0 | Specification + **scope freeze** | 0 | 2 | No | all |
| 1 | Repository and configuration | 1 | 2 | No | P2 |
| 2 | Environment abstraction + first 3 environments + cost-accounting interface | 2 | 6 | No | P2, P3 |
| 3 | Curator simulator and oracles | 3 | 8 | No | P1 |
| 4 | Baselines in simulator: Uniform / Static / LP | (part of 5, 7) | 3 | No | P1 |
| 5 | Discounted-UCB scheduler (Curator v0) + status logic | 5 (signals from 4) | 7 | No | P1 |
| 6 | Matched-compute simulation experiments (sweeps + scenario suite) | (part of 5) | 3 | No | P1 |
| 7 | Cost measurement + GRPO + TRL integration + static GRPO baseline | 6, 7, 8 | 18 | Yes | P2 |
| 8 | Held-out calibration + ROI engine | 9, 10 | 10 | Yes | P3 |
| | **MILESTONE M-1: MVP functionally complete** (Gates 6, 7) | | | | |
| 9 | Tier-1 comparison + 3 seeds + priority ablations | 11, 12 | 8 | Yes | P2, P3 |
| 10 | LOO validation (Tier 1) + sealed final test + plots, tables, report | 13, 15, 16 | 12 | Yes | P3 |
| | **MILESTONE M-2: MVP paper-ready** | | | | |
| | **Stretch below** | | | | |
| S-11 | Additional environments (MATH, MBPP, K&K, tool-use, too-hard) + Tier 2 pilots | (part of 11) | 8 | Yes | P2 |
| S-12 | Alternative schedulers: SW-UCB / EXP3 / DTS | (part of 5) | 4 | No | P1 |
| S-13 | SEC-style baseline | (part of 11) | 3 | Yes | P1 |
| S-14 | Large ablation grid | 12 | 6 | Yes | P3 |
| S-15 | verl adapter | (section G.6) | 4 | Yes | P2 |
| S-16 | Three-seed Tier-2 main experiments (1.5B) + full dashboard | 14 + viz | 9 | Yes | all |

| | **MVP subtotal** | **MVP + stretch total** |
|---|---|---|
| Person-days | **≈ 79** | ≈ 113 |

**Critical path (v2):** Phase 3 → 5 → 6 (Gate 2) → 7 → 8 (M-1) → 9 → 10 (M-2). Phases 4, and the cost-attribution part of 7, can run in parallel with Phase 3 once Phase 2's types exist.

**Note:** the v1 phase numbers are still used inside Part P, the defect table, and the gates table (with explicit labels). When a task there references e.g. "Phase 7: Static-mixture GRPO baseline", that is v1 Phase 7, inside v2 Phase 7.

### A.4 Team ownership and calendar (v2.0)

Ownership is by **system boundary**, so no two people write pieces that fail to integrate. Every teammate reviews the experimental protocol together.

| Person | Boundary | Owns |
|---|---|---|
| **P1 — Curator / Algorithm** | L1 pure + simulator | scheduler, signal engine, UCB variants, mixture map, simulator, oracles, scheduler tuning |
| **P2 — RL / Training** | L2 adapters | GRPO, TRL integration, environment interface, rollout collection, sandboxed code execution, GPU cost measurement, checkpoints/resume |
| **P3 — Evaluation / Research** | L3 + docs | calibration, α/β fit, credit rules (C1/C2), ROI engine, LOO, baselines' evaluation, statistical analysis, plots/tables, sealed-test protocol, paper |

| Week | Phases | Gate/Milestone | Notes |
|---|---|---|---|
| 1 | 0, 1 | Gate 0 | All three sign `spec-v1`; fix proposal references (S-15) and dashboard/verl wording (A0.5) |
| 2 | 2 | Gate 1 (3 envs) | P2 leads environments; P3 reviews split protocol |
| 3–4 | 3 | Gate 1 | **Buffer week 4 for the simulator** — it gates everything below |
| 4–5 | 4, 5 | Milestone M-0 (v0 sanity) | P1 lead, P3 starts calibration design documents |
| 6 | 6 | **Gate 2** | Scheduler validated off-GPU. No GPU work before this passes |
| 7 | 7a (cost + TRL spike) | — | P2; D-UCB variants (stretch S-12) may run if P1 free |
| 8 | 7b (GRPO baseline + integration) | Gates 3, 4, 5 | Integration is the main engineering risk |
| 9 | 8 | Gates 6, 7 | **Buffer week 9 for calibration + ROI** |
| 10 | 9 | Gate 8 | Tier-1 comparison runs; stretch S-11 may start in parallel |
| 11 | 10 | Gate 9 + freeze gate | LOO + sealed test + plots; freeze before any remaining expensive runs |
| 12 | 10 | **Milestone M-2: MVP paper-ready** | Report + checklist (Part U) |
| ~13+ | S-11..S-16 | Gates 1B/10 stretch | Only if MVP is done; anything not reached is reported as future work |

**Effort budget rule:** if any phase slips more than one week behind this calendar, the team cuts the next stretch item in the order S-16 → S-13 → S-12 → S-15, never an MVP item.

---

# PART B: Complete system architecture

## B.1 Components

| Component | Responsibility | Layer |
|---|---|---|
| Environment Registry | Maps names to environment objects, split manifests, and metadata | L2 adapter |
| Environment | Produces prompts, verifies completions, computes rewards (Part D) | L2 |
| Trainer Hook (`trainer/`) | Draws prompts by w_t, records rollouts, drives rounds, enforces budget | L2 |
| GRPO trainer (TRL) | Rollouts, advantages, LoRA updates. Untouched | external |
| Cost Meter (`cost/`) | Measures spans, attributes to environments, keeps the ledger | L2 |
| Signal Engine (`signals/`) | Pass rate, LP, SR, status, proxy reward per environment | L1 pure |
| Curator Scheduler (`scheduler/`) | D-UCB scores → softmax mixture, exploration floor, status constraints | L1 pure |
| Calibration (`calibration/`) | Calibration benchmark, evaluator, α/β fit, credit assignment | L1 + L2 |
| ROI Engine (`roi/`) | Per-environment gain, cost, ROI, confidence intervals, leaderboard | L1 pure |
| Evaluation (`evaluation/`) | Metrics, curves, LOO analysis, sealed final test | L3 |
| Simulator (`simulator/`) | Synthetic world with known optimum; trace replay | L2 |

## B.2 Layering rules (enforced by `tests/architecture/test_import_rules.py`)

- **L0 `core`:** dataclasses, config, seeding, logging. Imports nothing from the project.
- **L1 pure algorithms** (`signals`, `scheduler`, `calibration.regression/credit`, `roi`): import only `core`, numpy, and scipy. MUST NOT import torch, transformers, trl, datasets, `envs`, `trainer` or `evaluation.final_test`.
- **L2 adapters** (`envs`, `cost`, `trainer`, `simulator`): import `core`. Only `trainer` may import TRL, and only inside `trl_adapter.py`.
- **L3** (`evaluation`, `experiments`): may import everything except that nothing in L1 or L2 may import `evaluation.final_test` or the sealed-test loader.

## B.3 Data contracts (all in `core/types.py`)

| Type | Key fields |
|---|---|
| `Prompt` | `prompt_id` (stable string), `env_id`, `split`, `messages`, `reference` (verifier-only), `meta` |
| `Verdict` | `success` (bool), `score` ∈ [0,1], `parse_ok`, `verifier_seconds`, `info` |
| `RolloutGroup` | `env_id`, `prompt_id`, `n` (=G), `prompt_tokens`, `completion_tokens[]`, `scores[]`, `successes[]`, `verifier_seconds` |
| `EnvRoundObs` | `env_id`, `n_prompts`, `n_rollouts`, `k_success`, `n_groups_mixed`, `sum_score`, `sum_score_sq`, `prompt_tokens`, `completion_tokens`, `verifier_seconds`, `gpu_seconds`, `cost_usd` |
| `RoundObservation` | `round`, `steps`, `per_env: dict[str, EnvRoundObs]`, `weights_used`, `round_cost_usd`, `overhead_usd`, `budget_remaining_usd` |
| `CalibrationObservation` | `window_k`, `round`, `score_total`, `score_by_domain`, `se_total`, `se_by_domain`, `n_items`, `eval_cost_usd`, `exposure_by_env` (compute mass W_{j,k} since the last calibration), `window_cost_usd` |
| `MixtureDecision` | `round`, `weights`, `ucb_scores`, `mu_hat`, `bonus`, `statuses`, `intents`, `quotas`, `rng_hash` |
| `RoiRecord` | Fields listed in Part I.6 |

The simulator, the trace replayer, and the real trainer all produce the same `RoundObservation`. That is what lets the scheduler be validated off-GPU.

## B.4 Round loop (what happens, in order)

1. `Curator.select_mixture()` returns w_t (Part E).
2. `CuratorSampler` converts w_t into per-environment prompt quotas for the next R optimizer steps and draws prompts.
3. TRL runs R GRPO steps. The reward function wrapper routes each completion to its environment's verifier by the `env_id` column, times the verifier, and appends `RolloutGroup` records to the `RolloutLedger`.
4. The adapter records measured spans (generation, scoring, update) for each step. The Cost Meter attributes them to environments (Part H).
5. At the round boundary `CuratorCallback` builds `RoundObservation`, calls `SignalEngine.update()`, then `Curator.update_observation()`.
6. Every K rounds the callback freezes the checkpoint and runs `CalibrationEvaluator`. The evaluation cost is charged. Then `Curator.update_calibration()` runs.
7. The callback checks the budget ledger and sets `control.should_training_stop` when cumulative cost reaches B.
8. Logs and checkpoints are written atomically.

---

# PART C: Repository structure

The Python package is named `curator_rl`. This avoids a name collision with Bespoke Labs' existing `curator` library if either is ever installed in the same environment.

```
curator-rl/
├── pyproject.toml          project metadata, extras: dev, sim, train
├── uv.lock                 pinned dependency lock (or requirements.lock)
├── README.md  Makefile  .gitignore  .pre-commit-config.yaml
├── docs/                   SPEC.md, DECISIONS.md, COST_MODEL.md, EXPERIMENT_PROTOCOL.md, ANALYSIS_PLAN.md
├── configs/
│   ├── base.yaml           every scientific hyperparameter, with its documented default
│   ├── env/                one file per environment
│   ├── scheduler/          curator_v0, curator_full, uniform, static, lp, ucb, ...
│   ├── trainer/            qwen05b_lora, qwen15b_lora
│   ├── sim/                scenario files
│   └── experiment/         smoke, tier0, tier1, tier2, loo, ablations
├── src/curator_rl/
│   ├── core/               types, config, seeding, runmeta, jsonl, paths, atomic io
│   ├── envs/               base, registry, splits, gsm8k, math_env, countdown, mbpp, knights_knaves, toolcall, noisy, toohard, sandbox
│   ├── signals/            passrate, progress, richness, status, proxy, engine
│   ├── scheduler/          base, mixture, ducb, swucb, exp3, dts, curator (facade), baselines/{uniform,static,lp,ucb,sec}
│   ├── cost/               meter, ledger, gpu_stats, model
│   ├── calibration/        benchmark, evaluator, regression, credit
│   ├── roi/                engine, bootstrap, leaderboard
│   ├── trainer/            trl_adapter, sampler, callback, reward_wrapper, round_controller, checkpoint
│   ├── simulator/          world, scenarios, oracle, harness, replay
│   ├── evaluation/         metrics, curves, stats, loo, guard, final_test
│   └── viz/                one module per figure family
├── data/
│   ├── raw/                downloaded originals (gitignored)
│   ├── processed/          normalised JSONL per env and split (gitignored)
│   ├── manifests/          committed: ID lists and SHA-256 per split
│   └── test_sealed/        gitignored; read only through evaluation.guard
├── experiments/            run_sim.py, run_pilot.py, run_main.py, run_loo.py, sweeps/, analysis/
├── scripts/                download_data.py, build_splits.py, cost_calibration.py, env_info.py
├── tests/                  unit/, contract/, integration/, architecture/, fixtures/
├── checkpoints/ logs/      gitignored run outputs (runs/{run_id}/...)
├── reports/                figures/, tables/, analysis plan, final report
└── notebooks/              exploration only. Nothing imports from here
```

| Directory | Purpose |
|---|---|
| `docs/` | The frozen specification and decision log. Code follows the docs, not the other way round |
| `configs/` | Every hyperparameter. A run is fully described by a config file plus overrides |
| `core/` | Shared types and infrastructure with no scientific logic |
| `envs/` | Data → prompts → verified rewards. No scheduler logic |
| `signals/` | Turns observations into pass rate, LP, SR, status, and proxy reward |
| `scheduler/` | D-UCB, softmax mixture, floor, and all baselines behind one `BaseScheduler` |
| `cost/` | The only place that reads clocks or GPU counters |
| `calibration/` | Calibration benchmark evaluator, α/β fitting, credit rules |
| `roi/` | ROI estimation, bootstrap intervals, leaderboard |
| `trainer/` | The only code that touches TRL |
| `simulator/` | Synthetic worlds, oracles, trace replay |
| `evaluation/` | Metrics, statistics, LOO analysis, sealed test access |
| `data/manifests/` | Makes splits deterministic and reviewable |
| `tests/architecture/` | Fails the build if a layering or sealing rule is broken |

---

# PART D: Data and environment architecture

## D.1 Common environment interface

The interface is frozen before any environment is written (Phase 2, first task). The method names follow your brief; the contracts are what the tests check.

| Method | Contract |
|---|---|
| `generate_prompt(split, rng) -> Prompt` | Deterministic given `(split, rng state)`. Never returns an item from another split |
| `generate_batch(split, n, rng, exclude=None) -> list[Prompt]` | Sampling without replacement within a pass over a finite split. Fresh instances for procedural environments |
| `evaluate_response(prompt, response) -> Verdict` | Verification only. Pure, thread-safe, has a timeout, and times itself (`verifier_seconds`) |
| `compute_reward(verdict) -> float` | Reward shaping lives here, separate from verification. Range [0,1] unless the spec says otherwise |
| `estimate_cost() -> CostPrior` | A prior for cold start only. It is never used as a measured cost |
| `metadata() -> dict` | Dataset version, split sizes, manifest hash, difficulty parameters |
| `split_ids(split) -> list[str]` | Used by leakage tests |

**Who produces what.** Every quantity Curator needs comes from one owner:

| Quantity | Produced by |
|---|---|
| Success, reward, verifier time, parse failure | Environment (`Verdict`) |
| Prompt tokens, completion tokens, rollout count | Trainer adapter / reward wrapper |
| GPU-seconds, dollars | Cost Meter |
| Pass rate, learning progress, signal richness, status, proxy reward | Signal Engine |
| Weights, UCB scores | Scheduler |

An environment never computes learning progress, and the scheduler never reads a clock.

## D.2 Datasets, preprocessing and rewards

All dataset identifiers below are **[verify]** at download time. Record the exact revision hash in `data/manifests/`.

**GSM8K** (`openai/gsm8k`, config `main`; about 7.5k train, 1.3k test)
- Preprocess: take the final answer after `####`, normalise numerics (strip commas and units, exact rational compare). Use a fixed prompt template asking for the final answer in a tagged span.
- Reward: 1 if the parsed answer equals the reference, else 0. Parse failure is recorded separately.
- Splits: official test is sealed. Official train is hash-split into train, calib (300) and dev (300).

**MATH subset** (Hendrycks et al.; HF mirror or GitHub original **[verify]**; MATH-500 is a subset of the MATH test split)
- Train on levels 3–5 of the train split. Extract `\boxed{}` answers and compare with a normalising equivalence checker (library choice **[verify]**).
- Splits: train split partitioned into train, calib, dev. MATH-500 is the sealed test.
- Caveat: Qwen2.5 pretraining may overlap GSM8K and MATH. Report procedural environments separately as a contamination-free signal.

**Countdown** (Reasoning Gym `countdown` task **[verify name]**, or a TinyZero-style generator)
- Reward: the expression uses only the given numbers, each at most once, and evaluates to the target. Evaluation uses an AST-based safe evaluator, never `eval`.
- Splits: disjoint seed ranges (train < 1e7 ≤ calib < dev < 2e7 ≤ test) plus a content-hash disjointness check.

**MBPP** (`google-research-datasets/mbpp`, full config, 974 tasks **[verify]**)
- Prompt shows the task text and one example assertion. Reward is 1 if all hidden assertions pass in a sandboxed subprocess (timeout, memory limit, no network).
- Splits: official train/validation/test map to train+dev / calib / sealed test. The calib slice is small (90), so its SE is large. Pool it into the aggregate score.
- Hacking risk: special-casing the shown assertion. Mitigation: grade on assertions not shown in the prompt.

**Knights and Knaves** (Reasoning Gym `knights_knaves` **[verify name]**, or the Xie et al. generator)
- Procedural, with difficulty = number of inhabitants. Exact-match verifier on the role assignment. Splits by seed ranges.

**Tool use (function calling)**
- Training environment: a synthetic, deterministic function-calling generator (calculator, unit conversion, date arithmetic, in-memory lookup API). Reward = valid call format AND correct function and arguments AND correct final result.
- BFCL is evaluation-oriented, so it is used as an external sealed test for this domain (subject to **[verify]** format and licence), not as training data.

**Noisy-reward (sanity)**
- A wrapper over the GSM8K train partition. Modes: `random` (reward ~ Bernoulli(q), independent of output) and `flip(p)` (true reward flipped with probability p). The true success is logged privately for diagnosis and is never exposed to the scheduler.

**Too-hard (sanity)**
- MATH level-5 problems filtered to those where the base model has pass@16 = 0, using the train partition only. The filter run is logged but not charged to any budget. The filter is re-run for each model size.

## D.3 Split protocol and leakage prevention

- **Four splits:** `train`, `calib`, `dev`, `test`. Calib drives the scheduler online. Dev is for tuning thresholds and hyperparameters and for validation curves. Test is sealed.
- **Assignment for finite datasets:** `u = int(sha256(salt + dataset + problem_id)[:8], 16) / 2**32`, thresholded into splits after the official test items are set aside. The salt lives in `configs/base.yaml` and is frozen at Gate 0.
- **Manifests:** `data/manifests/{env}_{split}.txt` plus a `MANIFEST.sha256`, committed to Git. `scripts/build_splits.py` is idempotent.
- **Sealed test:** `data/test_sealed/` is read-only. Access goes only through `evaluation.guard.SealedTest.open()`, which requires `CURATOR_FINAL_EVAL=1`, a frozen-config hash file for the experiment, and writes an entry to `reports/test_access_log.jsonl`. A second open for the same (checkpoint hash, split) fails.
- **Leakage tests** (Gate 1):
  1. Pairwise disjointness of split ID sets.
  2. Near-duplicate check (normalised text, 8-gram MinHash, Jaccard ≥ 0.8) between train and calib/dev/test, with duplicates removed from train.
  3. Disjoint seed ranges and content hashes for procedural environments.
  4. Static test: no L1 or L2 module imports `evaluation.final_test` or the sealed loader.
  5. Config validator: any `scheduler.*` or `calibration.*` setting that names `split: test` is rejected.

## D.4 Batching and sampling

- A round has `M = R × P` prompts (R optimizer steps per round, P prompts per step). Each prompt gets G rollouts, all from the same environment.
- Per-environment quotas: `m_i = round(M·w_i)` using largest-remainder rounding with a seeded tie-break. `stochastic` rounding is an option.
- Prompts are drawn without replacement within a pass over each finite train split, then reshuffled. The number of passes is logged.
- Each optimizer step's batch interleaves environments round-robin, so step batches are mixed and the quota is met per round, not per step.
- Per-environment `max_prompt_tokens` and `max_completion_tokens` come from config. Truncated completions count as failures and are recorded as `trunc_rate`.

## D.5 Per-environment contract tests (Gate 1)

1. Determinism: the same seed gives identical prompts.
2. Gold answers (when available) score `success = True` for at least 99%. Garbage and empty strings score `False` for at least 99%.
3. Timeouts work: an infinite-loop code completion returns within the timeout and is marked failed.
4. Splits are disjoint (D.3).
5. The noisy environment's success rate is within 0.02 of its design value. The too-hard environment's base-model pass rate is at most 0.02.
6. Throughput and verifier seconds are measured per environment and stored in `reports/env_profiles/`.

---

# PART E: Curator mathematical implementation

## E.1 Frozen notation

| Math | Config / code name | Meaning |
|---|---|---|
| i, N | `env_id`, `n_envs` | Environment index and count |
| t, `max_rounds` | `round_t` | Current round (R optimizer steps) and its limit |
| R, P, G, M | `steps_per_round`, `prompts_per_step`, `group_size` | Round and batch shape; M = R·P |
| K | `calib.interval_rounds` | Rounds between calibration evaluations |
| p̂_i, LP_i, SR_i | `pass_rate`, `lp`, `richness` | Signals |
| ĉ_i, c̃_i | `unit_cost`, `unit_cost_norm` | Measured dollars per prompt; divided by the mean across environments |
| r̃_i, r̄_i | `proxy_raw`, `proxy_unit` | Proxy reward before and after mapping to [0,1] |
| γ | `scheduler.gamma` | Bandit forgetting factor. The MDP discount is 1 |
| κ | `scheduler.exploration_coef` | Exploration coefficient (called c in Part 2) |
| τ, ε | `scheduler.tau`, `scheduler.epsilon` | Softmax temperature; exploration floor ε/N |
| α, β | `proxy.alpha`, `proxy.beta` | Proxy weights, initialised from a prior and then fitted |

## E.2 Observation processing

For environment i in round t the trainer reports `k` successes out of `n` rollouts, `m` prompt groups, and per-group pass-rate sums (`sum_group_rate`, `sum_group_rate_sq`). The Signal Engine keeps discounted pooled counts with a signal discount λ (separate from the bandit γ):

```
S_i(t) = λ·S_i(t-1) + k_i,t        N_i(t) = λ·N_i(t-1) + n_i,t
p̂_i(t) = (S_i + a0) / (N_i + a0 + b0)        Beta(a0, b0) prior, default (1, 1)
```

The standard error uses the prompt group as the unit, not the rollout, because rollouts within a group are correlated. Using rollouts would make LP look more significant than it is.

## E.3 Signals

Every formula below is selected by a config key. Nothing is hard-coded.

1. **Pass rate** p̂_i as above, with a Beta credible interval [p_lo, p_hi].
2. **Learning progress.** Four candidates:
   - **LP-A (fast minus slow):** p̂ with discount λ_f ≈ 0.7 minus p̂ with λ_s ≈ 0.95. The standard error is conservative because the windows overlap.
   - **LP-B (slope):** weighted least-squares slope of per-round p̂ over the last W rounds, weights ∝ n. Report z = slope / SE.
   - **LP-C (non-overlapping windows):** mean of the last W_s rounds minus the mean of the previous W_s rounds, with an independent-samples z-test.
   - **LP-D (advantage based):** mean absolute advantage from GRPO, as in SEC. It needs trainer internals, so it is used only in the SEC baseline.
   - **Selection protocol (Phase 4, on the simulator):** for each estimator, measure (a) rank correlation with the true skill velocity, (b) sign accuracy when the velocity is large, (c) lag, and (d) the false-positive rate of "significant LP" when true velocity is zero (noisy and saturated environments), under realistic sampling noise. Choose the estimator with the best correlation subject to false-positive rate ≤ 5% and lag ≤ 5 rounds. Fall back to LP-A.
   - **Sign handling:** `lp_use: signed | positive | abs`. Curator defaults to signed, clipped to [−L, L]. The learning-progress baseline uses `abs`, as in Graves and TSCL.
3. **Signal richness.** The fraction of prompt groups with an informative pass rate. `richness.mode`:
   - `mixed`: 0 < k < G (for G ≤ 9, the 0.1–0.9 band reduces to this).
   - `band`: lo ≤ k/G ≤ hi, with G ≥ 16 recommended.
   - `variance`: group reward std > σ_min, for continuous rewards.
4. **Reward statistics:** mean and std of scores, truncation rate, parse-failure rate.
5. **Cost:** ĉ_i from the Cost Meter (Part H). c̃_i = ĉ_i / mean_j ĉ_j.
6. **Saturation indicator:** p_lo ≥ p_sat. **Too-hard indicator:** p_hi ≤ p_hard and SR ≤ sr_hard.
7. **Token usage:** mean completion length and its trend (long completions drive cost).

## E.4 Proxy reward and normalisation (resolves S-5)

```
x_i   = α·LP'_i + β·SR_i            LP' = clip(LP / σ_LP, -L, L), σ_LP a running robust std across environments
r̃_i  = x_i / c̃_i^θc                θc = 1 (proposal), θc = 0 gives the "no cost normalisation" ablation
r̄_i  = clip((r̃_i − q_lo) / (q_hi − q_lo), 0, 1)     q_lo, q_hi = running discounted 5th and 95th percentiles
```

During warm-up, q_lo and q_hi come from config priors. Before calibration, α and β take fallback values (0.5, 0.5) on the normalised features. Part I.4 describes how they are fitted.

## E.5 Environment status classification (resolves S-1, S-3)

Inputs per environment: `n_groups_eff`, [p_lo, p_hi], p̂, z_LP, SR, and the calibration mismatch flag.

| Status | Enter when | Leave when |
|---|---|---|
| **S1 Unexplored** | `n_groups_eff` < `n_min` (default 64) or rounds seen < `r_min` (default 3) | both minimums are met |
| **S5 Unreliable** | the calibration engine flags proxy-vs-benchmark mismatch for ≥ q consecutive windows (default 2) | no mismatch for q' consecutive windows (default 2) |
| **S3 Saturated** | p_lo ≥ `p_sat` (0.85) and z_LP < `z_up` (2.0), for h consecutive rounds (default 3) | p̂ < `p_sat − h_p` (h_p = 0.05), or z_LP ≤ −`z_neg` (forgetting) |
| **S4 Too hard** | p_hi ≤ `p_hard` (0.10) and SR ≤ `sr_hard` (0.10) and z_LP < `z_up`, for h rounds | p̂ ≥ `p_hard + h_p`, or z_LP ≥ `z_up`, or SR > `sr_hard + h_sr` |
| **S2 Learning** | none of the above hold | any other status condition holds |

S2 carries a sub-label for display and constraints: **S2a (progressing)** when z_LP ≥ `z_up`, and **S2b (plateau)** when p̂ is mid-range and |z_LP| is small. A noisy-reward environment looks like S2b before calibration, because its richness is high and its LP is flat. That is exactly the failure the calibration step is supposed to catch. If it did not appear, hypothesis H3 would have nothing to test.

**Priority** is S1 → S5 → S3/S4 → S2. **Dwell:** a minimum of `D_min` rounds (default 5) in any status before leaving it, except leaving S1 and entering S5. **Smoothing:** discounted pooled counts and credible intervals. **Anti-oscillation metric:** `flip_rate` = transitions per environment per 100 rounds after warm-up. Target ≤ 5 under simulated noise. A unit test feeds p̂ sequences that oscillate inside the hysteresis band and asserts no transitions.

**How thresholds are chosen (no arbitrary values):**
1. Express each threshold as a statistical quantity (interval bound, z-score) with a conventional starting value.
2. Tune on the simulator, where the true status is known (true p and true velocity). Maximise macro-F1 over a grid of (p_sat, p_hard, z_up, h_p, D_min, h) subject to flip_rate ≤ 5. Use tuning seeds disjoint from evaluation seeds.
3. Validate on Tier 1 single-environment runs: the too-hard environment must reach S4, and the GSM8K environment must reach S3 once it saturates.
4. Freeze the values in `docs/DECISIONS.md` before the main experiments.

**Mapping to the Part 2 action menus** (`status_control: off | soft | hard`; default `soft`, to be chosen in Phase 12):

| Status | Constraint on w_t | Menu items in Part 2 |
|---|---|---|
| S1 | Exploration quota: w_i ≥ q_explore until exit; larger while `n_groups_eff` < `n_min`/2 | A1, A2, A3 |
| S2 | None; weights come from the UCB scores. The intent label is derived from Δw and uncertainty | B1, B2, B3 |
| S3 | Soft: UCB score × `m_sat` (0.5). Hard: cap at 2 × floor. Released weight is redistributed by normalisation | C1, C2, C3 |
| S4 | Soft: score × 0.5. Hard: cap at the floor. Redistribution is automatic | D1, D2, D3 |
| S5 | Shrink m̂ by `s_unrel` (0.5); request an extra calibration if the last one is older than K/2 rounds | E1, E2, E3 |

## E.6 Discounted UCB and the softmax mixture

UCB produces scores. The softmax is a separate step that turns scores into weights.

Observation exposure is u_{i,t} = m_{i,t}/M (the fraction of round-t prompts from environment i). `pull_unit: round | prompt` sets the scale and so the meaning of κ. For every environment with u > 0:

```
Ñ_i(t) = γ·Ñ_i(t-1) + u_i,t          Σ̃_i(t) = γ·Σ̃_i(t-1) + u_i,t · r̄_i,t
μ̂_i    = Σ̃_i / Ñ_i                   Ñ(t) = Σ_i Ñ_i(t)
s_i    = μ̂_i + κ · sqrt( ln(max(Ñ(t), e)) / Ñ_i )                      UCB score
z_i    = (s_i − mean(s)) / (std(s) + η)                                  score_norm: zscore | rank | none
w_i    = (1−ε) · softmax(z_i / τ) + ε/N                                  floor: w_i ≥ ε/N, Σ w_i = 1
```

- **Unexplored environments** (Ñ_i ≈ 0) are handled by the S1 exploration quota and warm-up, not by setting s_i = ∞, which would overflow the softmax.
- **Warm-up:** the first `warmup_rounds` (default 2N) use a uniform mixture to initialise costs and signals.
- **Status constraints** are applied after the base weights: apply multipliers or caps, redistribute excess to uncapped environments proportionally, then re-impose floors and quotas. Sum = 1 within 1e-12 is a property test.
- **Numerics:** clip z/τ to [−50, 50] and use log-sum-exp.
- **Variants** (after Phase 11, same API): sliding-window UCB, EXP3 (importance-weighted estimates), discounted Thompson sampling. Only discounted UCB is built first.

## E.7 Hyperparameters and how each is selected

| Config | Initial default | Selected by |
|---|---|---|
| `gamma` | 0.95 | Simulator grid {0.90, 0.95, 0.98} on tuning seeds, confirmed on dev |
| `exploration_coef` κ | 0.5 | Simulator grid |
| `tau` (on z-scores) | 1.0 | Grid {0.3, 0.5, 1, 2} |
| `epsilon` | 0.10 | Grid {0.05, 0.10, 0.20}; starvation tests |
| `warmup_rounds` | 2N | Simulator |
| `signal discounts` (λ, λ_f, λ_s) | 0.9, 0.7, 0.95 | LP estimator study (E.3) |
| `calib.interval_rounds` K | 5 | Overhead rule K ≥ c_eval / (ρ_max · c_round), with ρ_max = 0.10 |
| `steps_per_round` R | 5 | Phase 7 pilot |
| α, β | prior (0.5, 0.5), then fitted | Part I.4 |

Hyperparameters are tuned only on simulator tuning seeds and the `dev` split. Never on `test`, and the final-run configs are frozen with a hash before the final evaluation.

---

# PART F: Simulator implementation (Tier 0)

## F.1 Purpose and rules

The simulator validates the scheduler independently of LLM training. It MUST produce the same `RoundObservation` and `CalibrationObservation` types as the real system, and the scheduler MUST NOT be able to read any hidden simulator variable.

## F.2 World model

Each environment i has a latent skill s_i and a benchmark slice with the same domain.

```
p_i      = σ( a_i · (s_i − d_i) )                       true pass probability; a_i slope, d_i difficulty
g_i      = 4 · p_i · (1 − p_i)                           learnability, peaks at p = 0.5 like GRPO signal
s_j(t+1) = s_j(t) + R · η_j · Σ_i T_ji · w_i · g_i + ξ    T: transfer matrix (T_ii = 1; T_ji < 0 is interference)
cost/round = M · Σ_i w_i · c_i(t)                        so cheaper mixtures buy more rounds from a fixed budget
```

- **Observed rollouts:** for each of the m_i prompts, a per-prompt pass probability q ~ Beta with mean p_i and concentration κ_b, then k ~ Binomial(G, q). This gives realistic signal richness, not just a binomial count.
- **Benchmark:** S_d = mean over items of σ(a(s_d − d_d^bench)), observed with Binomial noise over n_b items per domain. S = Σ_d π_d S_d.
- **Cost:** c_i with log-normal noise and optional drift (response length grows as skill rises).

**Environment types** (each is a parameter setting, not separate code):

| Type | Setting | Tests |
|---|---|---|
| Standard | moderate d_i, positive η | basic allocation |
| Easy / saturating | low d_i | saturation handling |
| Delayed | η_i gated by σ(s_prereq − θ) | exploration; revival |
| Noisy | reported success ~ Bernoulli(q) independent of skill; η_i = 0; T_ji = 0 for all j | proxy misalignment |
| Too hard | d_i far above s_i; may unlock through T_{i,j} > 0 | zero-signal and revival |
| Interfering | T_ji < 0 | negative transfer |
| Drifting | d_i shifts abruptly at round t0 | value of discounting |
| Cost-drifting | c_i rises with s_i | cost estimation |

Parameters are drawn from distributions per seed, so the scheduler is not tuned to one hand-built world.

## F.3 Scenario suite

| ID | N | Contents | Primary question |
|---|---|---|---|
| S-A | 3 | easy, valuable-slow, noisy | Does it drop the noisy one and the saturated one? |
| S-B | 8 | the full proposal portfolio analogue, including noisy and too-hard | Realistic overall comparison |
| S-C | 4 | cost heterogeneity 1× to 8×, equal gains | Value of cost normalisation (H2) |
| S-D | 4 | transfer and interference matrix | Credit rules C1 vs C2 (H5) |
| S-E | 3 | abrupt regime change in one environment | Discounting vs stationary UCB (H4) |
| S-F | 3 | too-hard environment unlockable via transfer | Starvation and revival (floor) |
| S-G | 8 | evaluation-noise sweep (n_b, K) | Choice of K and set size |
| S-H | 5 | late-arriving environment | Cold start |

## F.4 Known optima (oracles)

- **O1 exact DP (N ≤ 3):** skills discretised to about 12 levels each, mixtures on a simplex grid with step 0.1, remaining budget discretised. Deterministic mean dynamics. Value = final benchmark score. This is the true optimum for the discretised problem.
- **O2 best static mixture:** grid or CMA-ES search over a fixed mixture, using true dynamics.
- **O3 myopic marginal-ROI oracle (any N):** at each round, allocate by the true m_i(s)/c_i with the equimarginal rule.
- **O4 references:** uniform and random.
- **Reported metric:** *oracle-gap closure* = (S_method − S_uniform) / (S_oracle − S_uniform), at equal budget.

## F.5 Harness and outputs

`run(scheduler_factory, scenario, seed) → logs` with budgets matched by the same ledger rule as the real system. Seeds: at least 50 per scenario for each method, in parallel on CPU. Outputs: per-round logs (same schema as real runs), final scores, oracle gap closure, status F1 against true status, junk-environment compute share, ROI rank correlation against true marginal ROI.

## F.6 Trace replay (Tier 1R)

Per-environment traces from real Tier 1 runs (pass counts, costs, benchmark deltas) are replayed against candidate schedulers.
- **Validity assumption (arm clock):** an environment's outcome depends only on its own number of pulls. This ignores transfer and interference.
- **Coverage requirement:** the logged runs must have used randomised mixtures wide enough to cover what the replayed scheduler will choose.
- **Use:** scheduler logic, hyperparameters, and regret. **Never** for transfer claims or leaderboard claims.

## F.7 Sim-to-real calibration

Fit η, T, a, d, and cost curves from the Tier 1 transfer pilot (single-environment training, all-slice evaluation). Re-run the scenario suite with the fitted values. The simulator is an algorithm testbed, not evidence about real models. Every headline claim is confirmed in Tier 2.

---

# PART G: GRPO integration

## G.1 Options considered

| Option | Idea | Verdict |
|---|---|---|
| **A** | A dynamic prompt stream (`IterableDataset` or sampler) that reads the current mixture from a shared `MixtureState`, which a callback updates at round boundaries | **Preferred**, if the pinned TRL version supports it with grouped generations |
| B | Re-create the dataset and call `train()` once per round | Risky: optimizer and scheduler state handling across calls |
| C | TRL's custom `rollout_func`, generating per environment | Not the default. Used in Tier 1 as an exact per-environment timing cross-check for Gate 5 |
| D | A minimal in-house GRPO loop | Fallback only, if the TRL hooks prove brittle |

The choice is made in a time-boxed spike at the start of Phase 7 and recorded in `docs/TRL_SPIKE.md`.

## G.2 What is known and what must be checked

Checked against current TRL documentation: reward functions receive `prompts`, `completions`, `completion_ids` and every dataset column as keyword arguments; `GRPOTrainer` accepts `callbacks`; it supports a custom `rollout_func` (extra returned fields are forwarded to reward functions); `train_dataset` may be a `Dataset` or an `IterableDataset`. The docs I read cover versions through v1.x, so **pin an exact version**.

**[verify in the spike]**
1. How the trainer repeats prompts for G generations when the dataset is an `IterableDataset`, and whether group contiguity is preserved.
2. Where generation and optimisation can be timed without touching private methods.
3. Whether `completion_ids` includes padding, for token counts.
4. Prefetch behaviour of the dataloader (weight-update lag).
5. Interaction with LoRA and, if used, vLLM.

All TRL-specific code lives in `trainer/trl_adapter.py` behind a `TrainerAdapter` protocol. If a private method must be wrapped, it is wrapped there and covered by a test that fails loudly on a version change.

## G.3 Hook sequence (your steps 1–10)

| Step | Action | Owner |
|---|---|---|
| 1 | Curator returns w_t | `CuratorCallback` at a round boundary (`global_step % R == 0`) writes it to `MixtureState` |
| 2 | Prompts sampled by w_t | `CuratorPromptStream` converts w_t to quotas (D.4) and yields prompts |
| 3 | GRPO rollouts | TRL (unchanged) |
| 4 | Rewards computed | `RewardWrapper` (single reward function, routes by `env_id`) |
| 5 | Training update | TRL (unchanged) |
| 6 | Per-environment statistics | `RolloutLedger` → `RoundObservation` |
| 7 | Cost measured | Adapter spans → Cost Meter (Part H) |
| 8 | Signals computed | `SignalEngine.update()` |
| 9 | Curator updated | `Curator.update_observation()` |
| 10 | Calibration every K rounds | `CuratorCallback` → `CalibrationEvaluator` → `Curator.update_calibration()` |

**Reward wrapper.** One function: `reward(prompts, completions, completion_ids, env_id, prompt_id, reference, **kw)`. It groups items by environment, runs CPU-bound verifiers in a process pool with timeouts, records every `Verdict` and token count, and appends `RolloutGroup` records to the ledger. It asserts that each prompt id appears exactly G times. Any format shaping is part of the environment's own reward and is applied identically in every method.

**Budget and checkpoints.** After each step the callback adds measured cost to the ledger. When cumulative charged cost ≥ B it sets `control.should_training_stop = True`. Checkpoints are written at round boundaries: TRL's own state (adapter, optimizer, scheduler, RNG) plus `curator_state.json` and the ledger, in one atomic step directory. Resume refuses to start unless both halves agree on the step number.

## G.4 Guards that Curator does not change GRPO correctness (Gate 4)

1. **Passthrough equivalence:** Curator with a fixed uniform mixture versus a plain TRL run on a shuffled uniform dataset. Pass = paired final scores and mean per-step reward within the pre-registered equivalence margins δ (Gate 4); two-sample KS on reward, loss and KL are reported as diagnostics, not as the equivalence claim. 3 seeds.
2. **Realised proportions:** realised environment shares match requested weights within ±2 percentage points over 100 rounds (chi-square p > 0.05).
3. **Group integrity:** every prompt's G completions come from one environment.
4. **No optimiser drift:** learning rate, KL coefficient, batch shape and clipping are identical across methods. The learning rate is constant after warmup (S-14).
5. **Weight-update lag:** at most one optimizer step between a new w_t and its first batch.

## G.5 Initial training configuration (all values are config, set in the Phase 7 pilot)

LoRA on all linear layers (rank 16 as a starting point), G = 8, P = 16 prompts per step, max completion length 512 (math/logic) up to 1024 (code/tool use), constant learning rate with warmup, KL coefficient from the pilot, bf16 where supported (fp16 on T4). Tier 1 uses plain HF generation first so timing is simple and exact. vLLM is considered only after Gate 5 because its batching changes cost attribution.

## G.6 verl

A `verl_adapter.py` implementing the same `TrainerAdapter` protocol is Stretch (S-15): it comes only after the MVP (v2 Milestone M-2), as future work. Nothing in L1 changes.

---

# PART H: Cost measurement

Cost is central, so it is never estimated from FLOPs or from `estimate_cost()` priors. Everything below is measured.

## H.1 What is measured

Wall-clock time; GPU-seconds; GPU utilisation and memory (sampled at 1 Hz with NVML where available); prompt tokens; completion tokens; rollout count; verifier / unit-test / tool-execution time; dollars (or normalised units).

## H.2 Spans: exactly where timing starts and stops

All clocks are `time.perf_counter_ns()`, with `torch.cuda.synchronize()` immediately before reading the clock at every span boundary.

| Span | Starts | Stops |
|---|---|---|
| `GEN` | Immediately before the first generate call of the step | After the last completion is returned and the device is synchronised |
| `SCORE` | Entry to the reward wrapper | Exit from the reward wrapper. Per-item verifier seconds are recorded separately |
| `UPDATE` | After rewards are available. Includes reference-model log-probs, policy forward/backward and the optimiser step | After the optimiser step and synchronisation |
| `EVAL` | Entry to the evaluator | Exit, with synchronisation. Calibration and reporting evaluations are separate categories |
| `OVERHEAD` | Entry to Curator code (signals, scheduler, ROI, logging) | Exit |

Every wall-clock second between step begin and step end belongs to exactly one span or to `unattributed`. `unattributed` MUST stay below 3%.

## H.3 Cost model

```
gpu_seconds = wall_seconds × n_gpus                    reserved-GPU model
usd         = gpu_seconds × usd_per_gpu_hour / 3600
```

If real prices are unavailable, set `usd_per_gpu_hour = 1.0` and report "GPU-units". Verifier time counts as GPU time by default (the GPU is reserved while verification runs). `cost.count_cpu_verifier_as_gpu_time` can turn this off when verification is fully overlapped with generation. The choice goes into the report.

## H.4 Attribution to environments

Batches mix environments, so generation and update time cannot be read per environment directly.

- **SCORE:** exact. The wall time of parallel verification is split in proportion to per-item verifier seconds.
- **GEN:** split by a fitted share `a_g·completion_tokens_i + b_g·n_seqs_i + c_g·prompt_tokens_i`.
- **UPDATE:** split by `a_u·tokens_i + b_u·n_seqs_i`, over prompt + completion tokens in the training batch.
- **Reconciliation:** the shares are rescaled each step so the attributed total equals the measured total. The coefficients come from single-environment micro-benchmarks in Phase 6.
- **Validation (Gate 5):** per-environment attributed cost matches exact timing from separate single-environment generation calls (the `rollout_func` cross-check) within 10%.

## H.5 Unit cost used by the scheduler

ĉ_i is an exponential moving average of the attributed dollars per prompt (all G rollouts plus verification). It starts from the Phase 6 cost-calibration run, never from `estimate_cost()`. c̃_i = ĉ_i / mean_j ĉ_j.

## H.6 Budget ledger

| Category | Charged to B? |
|---|---|
| `train_gen`, `train_score`, `train_update` | Yes |
| `calib_eval` (scheduler-owned calibration evaluation) | Yes |
| `curator_overhead` | Yes |
| `report_eval` (validation curves on dev, identical protocol for every method) | No |
| `setup` (for example the too-hard filter run) | No, but logged |

Stop at the first optimizer step where cumulative charged cost ≥ B. The overshoot is logged and is at most one step. This rule applies to every method, so a method that is not Curator does not pay for evaluations it does not use, and Curator pays for every evaluation it uses.

## H.7 Validation protocol (Phase 6)

1. **Cost calibration run:** for each environment, 200 prompts × 3 repeats with the base model. Per-environment CV of unit cost must be below 10%.
2. **Warm-up exclusion:** drop the first 2 steps of any measurement.
3. **Drift monitor:** re-run an identical micro-batch every N steps and record its cost. Drift above 10% triggers a flag.
4. **Isolation:** GPU not shared; fixed CPU count for verifiers; clocks locked if permitted, otherwise recorded.
5. **Ordering sanity:** tool-use and code should cost more per prompt than GSM8K. If not, investigate before trusting any number.

## H.8 Known biases

Warm-up and compilation effects; response length changing during training (handled by the EMA and the drift monitor); GPU contention; verifier CPU variability; batching efficiency depending on batch composition (attribution by shares is an approximation, stated as a limitation); prefix caching if vLLM is used; hardware dependence (always report the GPU model, and treat normalised costs as relative).

---

# PART I: Calibration and ROI engine

## I.1 Calibration benchmark

- **Structure:** one slice per environment domain. About 200 items per slice, so the aggregate has about 1,000 items (SE ≈ 1.5 points at 40% accuracy, tighter for paired differences). The code slice may be smaller and noisier.
- **Protocol:** freeze the checkpoint, greedy decoding, max tokens equal to training, fixed item order. Items are paired across evaluations.
- **Cost:** charged to `calib_eval`. Choose K with K ≥ c_eval / (ρ_max · c_round), ρ_max = 0.10, so calibration costs at most about 10% of the budget.
- **Outputs:** S_k, S_{d,k}, and standard errors from a paired bootstrap over items.
- **Isolation:** no training environment may sample a calibration item. Enforced by split manifests and leakage tests.

## I.2 Critical examination of the compute-share credit

The proposal's calibrated reward is r_i^cal = ΔS · share_i / c_i. I recommend against using it as the main calibrated reward. Here is the analysis you asked for.

**1. What it assumes.**
- Gain is additive across environments, and each environment's contribution is proportional to the compute spent on it. That treats every unit of compute as equally productive, which contradicts the premise that environments differ in value.
- No transfer or interference between environments.
- ΔS is measured without noise and is caused by the current window's training only (no delayed effects).
- The benchmark weights the environments' skills equally.

**2. Limitations.**
- **It carries no information about which environment caused the gain.** Credit depends only on allocation. With cost-based shares, every environment active in a window receives the same ROI, ΔS / C_window. With weight-based shares, credit is proportional to w_i/c_i, which is just the current allocation divided by cost.
- **Feedback loop.** Allocation determines credit, and credit determines the next allocation. Whatever the scheduler starts favouring keeps being rewarded.
- **Noise.** There is no uncertainty estimate, and a noisy negative ΔS punishes whichever environment had the largest share.
- **Delayed effects.** Gains caused by earlier windows are attributed to the present allocation.
- **It is not causal.** It is confounded by saturation timing, prior training, transfer, and measurement noise.

**3. Should it be the main calibrated reward?** No.

**4. Allocation-based proxy.** Yes: it is kept under the name `share_credit` (rule C1) for three uses. It is the fallback before enough windows exist for regression. It is the baseline in ablation A4. And the window-level ratio ΔS / C_window is a useful global efficiency metric and a scale for the proxies.

**5. Leave-one-out.** LOO is stronger because it intervenes on the training distribution with a counterfactual run, instead of accounting for where compute went. It has its own limits (Part K). The online ROI leaderboard is an estimate from one run, and LOO validates it on a subset.

**Recommended replacement, which keeps the original intent** (benchmark gain credited to environments and divided by cost):

- **C2, windowed regression credit.** Model ΔS_k = Σ_j W_{j,k}·m_j + noise, where W_{j,k} is environment j's compute mass in window k and m_j its gain per unit mass. Fit by Bayesian ridge with prior m_j ~ N(m_proxy_j, σ²), where m_proxy_j = g_φ(LP_j, SR_j). Weights are 1/SE². Identifiability needs variation in mixtures across windows, which the floor ε, the warm-up with randomised mixtures, and the exploration bonus provide.
- **C2d, domain-aware variant.** Use per-slice gains: ΔS_{d,k} = Σ_j W_{j,k}·B_{dj} + noise, with prior mass on the diagonal (an environment mostly affects its own domain). Then m_j = Σ_d π_d·B_{dj}. This multiplies the number of observations by the number of domains.
- **Calibrated reward for the bandit:** r_j^cal = m̂_j (posterior mean) / c̃_j, mapped to [0,1] like the proxy. It has the same form as the proposal but data-driven credit.
- **Both C1 and C2 are always computed and logged**, so ablation A4 is a re-analysis, not a new run.

## I.3 Per-window quantities

ΔS_k = S_k − S_{k−1} and ΔS_{d,k}, with paired-bootstrap standard errors. Regression rows are weighted by 1/SE². Windows with |ΔS| below one SE are kept but carry little weight.

## I.4 α and β calibration

The method is chosen empirically in the simulator before it is trusted on real runs.

| Aspect | Specification |
|---|---|
| Training data | Rows from completed windows. Default `calib.target: domain` uses one row per (window, environment): own-slice gain y_{d,k} against own signals. `global` uses one row per window. The domain mode gives N× more rows |
| Target | y = gain over the window (per slice or total), divided by SE (weighted least squares) |
| Features | x = (exposure-weighted LP′, exposure-weighted SR), z-scored with running statistics. Cost normalisation is applied after calibration, so α and β describe gain, not cost |
| Model | min Σ ω_k·(y − φᵀx)² + ν‖φ − φ₀‖², φ ≥ 0, solved as bounded least squares. ω_k = discount^(age) |
| Regularisation | ν chosen by leave-one-window-out CV over a grid. Prior φ₀ is scaled so mean predicted gain matches mean observed gain on the first windows |
| Update frequency | Every window, once at least `k_min` windows (default 5) or 20 rows exist |
| Trust region | |Δφ|/|φ| ≤ 25% per update |
| Fallback | φ₀ = (0.5, 0.5) on normalised features before the first fit |
| Safeguards | Non-negativity; CV error compared with the null model φ₀ (switch back if worse); R² and sign stability logged; if CV is poor for L consecutive fits, set `calibration_state = unreliable` and use benchmark-only credit |

Estimators compared in the simulator: OLS, ridge-to-prior (above), non-negative ridge, and recursive least squares with forgetting. Selection criteria: out-of-window MSE, coefficient variance across seeds, and sign consistency. The tuning seeds are disjoint from the evaluation seeds.

## I.5 S5 (unreliable) detection

For each environment, standardise the residual between own-slice gain and proxy-implied gain. If the mean standardised residual over the last q windows is below −z_mis (default 2.0), the proxy overestimates what the benchmark shows. Set S5 and shrink m̂. Underestimation is logged but does not set S5.

## I.6 ROI engine

Per environment, maintained and logged:

| Field | Definition |
|---|---|
| `environment_id` | |
| `total_allocated_compute` | compute mass Σ_t u_{i,t}·M prompts, and dollars |
| `total_gpu_seconds`, `total_tokens` | from the ledger |
| `average_cost` | mean ĉ_i (dollars per prompt) |
| `proxy_reward`, `calibrated_reward` | means of r̄ and r^cal |
| `estimated_benchmark_gain` | Ĝ_i = Σ_k W_{i,k}·m̂_{i,k} |
| `estimated_ROI` | Ĝ_i / C_i, in benchmark points per dollar, with a bootstrap interval from resampling windows and refitting |
| `selection_count`, `current_weight`, `state` | |

**Reconciliation:** Σ_i Ĝ_i + residual = Σ_k ΔS_k. The residual is reported. For C1 it is zero by construction.

**Leaderboard:** Environment | Compute | Estimated gain | ROI [CI] | State, built only from measured data. The wording in all outputs is "estimated marginal contribution under the mixtures actually run", not a causal effect.

---

# PART J: Baseline implementation

All baselines are `BaseScheduler` subclasses chosen by config. They share the trainer, the budget ledger, the signals and the seeds.

## J.1 Implementation order

| # | Baseline | Rule | Purpose |
|---|---|---|---|
| 1 | **Uniform** | w_i = 1/N | The default practice. Also validates the static GRPO pipeline (Gate 3) |
| 2 | **Static mixture** | Fixed weights set before training and recorded in the analysis plan. Size-proportional is a special case (weights ∝ nominal dataset size) | Manual practice |
| 3 | **Random mixture** | Dirichlet(1) redrawn each round | Sanity floor |
| 4 | **Learning-progress curriculum** | Boltzmann over |LP| from the same Signal Engine, with the same floor. Cost-blind | Graves / TSCL style. Tests whether cost awareness adds anything |
| 5 | **Standard UCB** | UCB1 on the same proxy reward, but no discount, no cost normalisation, no calibration. Same softmax map for the action | Isolates the value of discounting, cost and calibration |
| 6 | **Curator v0 ("no calibration")** | Proxy + cost + discounted UCB | First Curator, built in Phase 5 |
| 7 | **Curator no cost** | θ_c = 0, calibration on | Ablation A |
| 8 | **Full Curator** | Everything on | Main method |
| 9 | **SEC-style bandit** | Absolute-advantage reward (LP-D) with a TD(0) bandit | Last, only if trainer advantage statistics are accessible |

**Fair tuning:** every method gets the same tuning budget on simulator tuning seeds and the `dev` split (for example 8 configurations in the simulator and 4 on Tier 1 dev). Static-mixture weights are fixed in advance.

---

# PART K: Leave-one-out validation

This is called validated contribution (attribution), not causal inference. The design supports a counterfactual on the training distribution, but environments transfer, interfere and substitute for one another.

## K.1 Design

For each environment i in the chosen subset:
1. Run with all environments (the full run).
2. Remove environment i, keep the total budget B, and repeat.
3. Contribution_i = Score_full − Score_without_i, paired by seed, with the same initial checkpoint and data seeds.
4. Contribution per dollar_i = Contribution_i / C_i (full run).

**Two variants.**
- **L-C (Curator base):** rerun Curator without i. The scheduler reallocates freely, so the result is opportunity-cost contribution. This is the primary check of Curator's own leaderboard.
- **L-U (uniform base):** remove i from a uniform mixture. A secondary check.

**Measured on the `dev` split**, not on `calib`, because calibration data drives the scheduler in L-C runs and would make the check circular. Report three numbers per environment: total macro score, own-domain slice, and other-domain aggregate (transfer).

**Subset selection** is fixed before looking at the final Curator leaderboard: stratify by the Tier 1 pilot ROI (high, medium, low), and always include the noisy and too-hard environments. Record it in the analysis plan.

## K.2 Statistics

Spearman ρ and Kendall τ between the Curator ROI (and estimated gain) ranking and the LOO ranking, per seed and pooled over seeds. Bootstrap intervals by resampling seeds. Top-k and bottom-k agreement. Specific check: do both sanity environments rank at the bottom? Success thresholds are written into the analysis plan before the final runs (suggested starting point: Kendall τ ≥ 0.4 and both sanity environments in the bottom three). With 5–8 environments, power is limited, and the report says so.

## K.3 Limitations (stated in the report)

1. Non-additivity and interactions between environments.
2. Redundancy: removing GSM8K may cost little because MATH covers it, so LOO understates its value.
3. In L-C the whole trajectory changes, not only one environment's compute.
4. Seed variance may exceed the contribution. Paired seeds help, and Tier 0 measures detection power.
5. Cost: one extra run per environment per seed.
6. It measures contribution to the final score at budget B, not marginal value per dollar at a given moment.
7. In Tier 0 the true counterfactual is known exactly, so LOO accuracy is checked there first (Gate 9).

---

# PART L: Experimental protocol

## L.1 Matched compute

The primary question is: for the same compute budget, does Curator give more useful improvement than static, uniform, learning-progress or standard-UCB allocation? So:
- Every method runs to the same budget B under the ledger rule (H.6).
- The headline plots are score versus GPU-dollar, not score versus step.
- Steps, rollouts and tokens differ between methods. They are reported as covariates, not hidden.

## L.2 Fairness table

| Item | Rule |
|---|---|
| Base model and initial checkpoint | Identical, including LoRA initialisation seed |
| Environment pool | Identical |
| Budget | Identical B, with calibration evaluations and Curator overhead charged to the method that uses them |
| Optimiser settings | Identical. Constant LR after warmup. No schedule depends on step count |
| Calibration, dev, test sets | Identical |
| Seeds | Same seed list for every method (paired) |
| Evaluation protocol | Identical decoding, items and metrics |
| Hyperparameter tuning | Equal tuning effort per method, on dev and simulator tuning seeds only |
| Unavoidable differences | Different step counts and token totals (reported); calibration cost exists only for methods that calibrate (charged) |

## L.3 Seeds and statistics

At least 3 seeds for every compared method pair at the run tier actually used (v2.0 MVP: 3 paired seeds at Tier 1; stretch: 5 in Tier 1, 3 in Tier 2, at least 50 in Tier 0). Report mean ± standard deviation and 95% bootstrap intervals, plus paired differences against uniform. No claim of improvement when intervals overlap. Because 3 seeds give little power, the simulator carries the statistical weight for algorithm-level claims and the real runs confirm them.

## L.4 Evaluation timeline

- Reporting evaluations (validation curves) on `dev` at fixed cost milestones (for example every B/20), uncharged, identical for every method.
- Calibration evaluations on `calib` every K rounds, only for methods that use them, charged.
- Final evaluation on `test` exactly once per final checkpoint.

## L.5 Pre-registration

`docs/ANALYSIS_PLAN.md` is committed before the final runs (Gate 10). It names the primary metric (final test score at budget B, macro-averaged, and compute-to-target), the secondary metrics, the LOO subset and thresholds, the baselines, the seed list, and the analysis scripts.

## L.6 Run hygiene

Crashed runs are rerun with the same seed. Diverged runs (KL blow-up, NaN) are reported, not dropped. The number of runs is fixed in advance. No seed is chosen after seeing results.

---

# PART M: Testing strategy

Unit tests come first. No GRPO training starts until Gate 2 and the relevant unit tests pass.

| # | Area | Key assertions |
|---|---|---|
| 1 | Environment interface | Determinism by seed; split membership; `evaluate_response` timeout; gold and garbage answers; contract suite per environment |
| 2 | Reward calculation | Known (prompt, response) → expected reward; noisy environment hits its design rate; range [0,1] |
| 3 | Learning progress | Known p̂ sequences give known LP signs; flat sequences give no significant LP at the intended false-positive rate; each estimator unit-tested |
| 4 | Signal richness | Hand-built groups give exact SR for `mixed`, `band` and `variance` modes |
| 5 | Cost measurement | Fake clock: spans sum to total; unattributed < 3%; attribution sums to measured total |
| 6 | UCB | Hand-computed scores for a small case; bonus shrinks as Ñ_i grows; discount lowers stale Ñ |
| 7 | Softmax mixture | Σw = 1 within 1e−12; w_i ≥ ε/N; monotone in score; no overflow at extreme scores |
| 8 | Exploration floor | After status caps and quotas, floors still hold |
| 9 | State classification | Each rule fires on crafted inputs; hysteresis and dwell stop flapping; priority order |
| 10 | Budget accounting | Ledger categories; stop rule; overshoot ≤ one step; calibration and overhead charged |
| 11 | Calibration | Regression recovers planted α, β in noiseless and noisy cases; trust region; fallback; switch back to prior |
| 12 | ROI calculation | Reconciliation identity; bootstrap interval coverage on synthetic data |
| 13 | Checkpointing | Save, load, and continue gives identical scheduler decisions; atomic writes survive an injected crash |
| 14 | Split isolation | Pairwise disjoint; near-duplicate test; sealed-test guard refuses without the env var; second open refused; import-rule test |
| 15 | Leave-one-out | In the simulator the pipeline recovers the known counterfactual contribution within tolerance |

**Property tests** (Hypothesis): weights always valid; scheduler is permutation-equivariant in environments; adding a constant to all scores changes nothing under z-scoring.

**Integration tests:**
1. Simulator end-to-end for every scenario with Curator and baselines.
2. Mock trainer: a fake model and fake environments drive the real callback, sampler and ledger.
3. Tiny-model GRPO smoke test (a few steps, tiny random Qwen-style model) in a GPU-marked test.
4. Kill-and-resume test at a random step.
5. Cost reconciliation test on a short real run.
6. Full pipeline dry run on a tiny model before Gate 10.

GPU tests are marked `@pytest.mark.gpu`, and default CI runs CPU only.

---

# PART N: Reproducibility strategy

- **Seeds:** `SeedManager(master_seed)` hands out named, independent streams (`data_order`, `sampler`, `sim_world`, `lora_init`, `bootstrap`, …) via `numpy.random.SeedSequence`. A stream depends on its name, not on call order.
- **Deterministic splits:** hash-based assignment with a frozen salt and committed manifests.
- **Configs:** pydantic-validated YAML. Scientific hyperparameters have no Python defaults, so a missing one is an error. The config hash is SHA-256 of canonical JSON.
- **Experiment IDs:** `{date}_{tier}_{method}_s{seed}_{git7}_{cfg6}`.
- **Checkpoints:** model adapter, optimiser state, RNG states, Curator state, ledger, environment statistics. Atomic write-then-rename. A `latest` pointer is updated last.
- **Resume:** scheduler decisions after resume are identical for identical observations. Model training after resume is statistically equivalent, not bit-identical (GPU nondeterminism), and the report says so.
- **Git:** commit SHA, dirty flag and a hash of the diff in `metadata.json`. A dirty tree is refused for final runs.
- **Dependencies:** `uv.lock` (or a pip-compile lock), exact TRL, transformers, peft and torch versions.
- **Hardware log:** GPU model, driver, CUDA, CPU count, RAM, nvidia-smi dump per run.

## N.1 Structured logging

`logs/rounds.jsonl` (append-only) and `env_rounds.parquet`, with a schema version field. Per round: `round`, `step`, `weights`, `statuses`, `pass_rates`, `lp`, `richness`, `proxy_raw`, `proxy_unit`, `ucb_scores`, `mu_hat`, `bonus`, `round_cost_usd`, `gpu_seconds`, `tokens`, `budget_remaining`, `calibration_score` (when run), `delta_s`, `estimated_roi`, `alpha`, `beta`, `calibration_state`, and the cost-ledger breakdown. Console logs are never the record.

---

# PART O: Hardware and compute plan

## O.1 Development platforms (v2.0)

| Activity | Platform |
|---|---|
| Day-to-day development | Windows + WSL2 (or Linux if available) |
| GPU experiments (Tier 1 pilots, GRPO, calibration evals) | Linux — WSL2 with native CUDA, Colab, or the university GPU server |
| CPU-only work | Simulator, unit tests, oracles, scheduler tests, ROI engine, evaluation scripts, plotting |

**Invariants (test-gated):**
1. The simulator and **all scheduler logic MUST run without a GPU**. CI runs CPU-only; a GPU dependency anywhere in L1 or the simulator fails the import-rule test.
2. Nothing in `src/` may hard-code a platform path; all paths come from config (`data.root`, `paths.runs_root`).
3. The MBPP sandbox and Countdown AST evaluator are cross-platform (subprocess with timeout on POSIX and Windows; no `bash`-only features). vLLM is Linux-first and stays deferred behind Gate 5 regardless.
4. GPU runs are only ever launched from Linux (WSL2/Colab/server). Windows is never a training platform.
5. Free/notebook GPU sessions require the resumability guarantees of Part N to be in place before the first real run (v1 Phase 7 checkpoints, kill-and-resume test).

## O.2 Tiers

| Tier | Hardware | Notes |
|---|---|---|
| 0 Simulator | Any CPU | Parallel over seeds. Hours in total |
| 1 Pilot and development | One 24 GB GPU (A10G, L4, RTX 4090 class) | Qwen2.5-0.5B-Instruct with LoRA. A T4 works but lacks bf16 and is slow. Free notebook sessions need resumability |
| 2 Main | One A100 40/80 GB, or two 24 GB GPUs | Qwen2.5-1.5B-Instruct with LoRA |

**Planning budget** (assumptions to replace with Phase 6 measurements). Assume a Tier 1 run costs B₁ = 1.5 GPU-hours and a Tier 2 run costs B₂ = 5 GPU-hours.

| Block | Runs | GPU-hours |
|---|---|---|
| Tier 1 pilots and cost calibration | about 15 short runs | ≈ 15 |
| Tier 1 methods | 8 methods × 3 seeds = 24 | 24 × 1.5 = 36 |
| Tier 1 ablations | 6 × 3 = 18 | 27 |
| Tier 1 LOO | 4 envs × 3 seeds = 12 | 18 |
| Tier 2 main | 7 methods × 3 seeds = 21 | 105 |
| Tier 2 LOO subset | 4 envs × 2 seeds = 8 | 40 |
| **Total** | | **≈ 240, plus 20% buffer ≈ 290** |

**MVP compute plan (≈ 40–60 GPU-hours, matches A0):** Tier 1 only — Uniform, Static, Learning-progress and Full Curator (4 × 3 = 12 runs) plus Tier-1 LOO (12 runs) and at most two priority ablations. The full 240–290-hour plan above belongs to the stretch tiers. **Scale-down levers:** shorter completions, a smaller calibration set with a larger K, fewer environments, vLLM after Gate 5.

---

# PART P: Phase-by-phase implementation roadmap

**v2.0 note.** The detailed phases below keep their **v1.0 numbering** so that the committed 17-item structure and all cross-references inside gates and tests stay valid. The v2.0 phases in Part A.3 are **groups over this list**:

| v2.0 Phase | Made of v1 Phases | Status | Gate / Milestone |
|---|---|---|---|
| 0 | 0 | MVP | Gate 0 |
| 1 | 1 | MVP | — |
| 2 | 2 | MVP | Gate 1 (3 envs) |
| 3 | 3 | MVP | — |
| 4 | baselines of 5 (Uniform / Static / LP) | MVP | — |
| 5 | 4 (signals, statuses) + rest of 5 (D-UCB, mixture map) | MVP | — |
| 6 | tuning + scenario sweeps of 5 | MVP | **Gate 2** |
| 7 | 6 + 7 + 8 | MVP | Gates 3, 4, 5 |
| 8 | 9 + 10 | MVP | Gates 6, 7 → **Milestone M-1** |
| 9 | 11 (Tier-1 subset) + 12 (priority) | MVP | Gate 8 |
| 10 | 13 (Tier 1) + 15 + 16 | MVP | Gate 9 → **Milestone M-2** |
| S-11 | 11 (full portfolio, part) | Stretch | Gate 1B |
| S-12 | SW-UCB/EXP3/DTS (part of 5) | Stretch | — |
| S-13 | SEC baseline | Stretch | — |
| S-14 | 12 (full grid) | Stretch | — |
| S-15 | — (G.6) | Stretch | — |
| S-16 | 14 (Tier 2) + viz | Stretch | Gate 10 final runs |

The original three changes below still apply with their v1 numbers: Phase 2 builds the interface plus only three environments (Rule 2), the remaining five arrive in v1 Phase 11 (v2 Phase 9 / S-11), and Gate 5 (cost) closes at the end of v1 Phase 8, because in-trainer attribution needs the integration.

**MVP discipline rule:** at every milestone, phases marked stretch in the table above are cut first on schedule slip. When v2 Milestone M-2 is reached, the project is complete and anything after it is reported as future work.

Every phase lists the same 17 items. Phases 0–1 are documentation and scaffolding; Phase 1 is specified in full at the end of this document.

---

## PHASE 0: Research specification and implementation contract

1. **Objective:** Freeze the notation, types, objective, accounting, split rules — and the **MVP/stretch scope** (A0.2).
2. **Why:** Fifteen specification defects (table at the top) would otherwise be resolved silently and differently by each teammate, and an unfrozen scope is the top schedule risk for a 3-person semester project.
3. **Tasks:** Write `SPEC.md` (notation E.1, interface D.1, contracts B.3, objective per S-10, accounting H.6, splits D.3). Open `DECISIONS.md` with one entry per defect. Choose the split salt. Fix the Part 2 references (S-15). **Freeze the A0.2 MVP/Stretch component table and the A.4 ownership/calendar table into `SPEC.md`. Record the dashboard/verl proposal wording corrections (A0.5) as DECISIONS entries and edit the proposal text accordingly.** Tag `spec-v1` after a three-person read-through.
4. **Files to create:** `docs/SPEC.md`, `docs/DECISIONS.md`, `docs/COST_MODEL.md` (skeleton), `docs/EXPERIMENT_PROTOCOL.md` (from Part L), `docs/GATES.md`.
5. **Files to modify:** none in the repo. The proposal document is corrected separately (A0.5).
6. **Classes/functions:** none. Types are defined in text only.
7. **Inputs:** proposal, Part 2, this roadmap.
8. **Outputs:** tag `spec-v1`; every defect marked resolved or accepted with a reason.
9. **Config parameters:** a list of every hyperparameter name and default (E.7), which becomes `base.yaml` in Phase 1.
10. **Unit tests:** none. A spec-lint check is added in Phase 1: every config key named in `SPEC.md` exists in `base.yaml`.
11. **Integration tests:** none.
12. **Expected result:** no symbol is used with two meanings; no open decisions.
13. **Failure conditions:** an unresolved decision; a notation clash.
14. **Debugging strategy:** each teammate re-derives the objective and the budget rule from the document alone and compares.
15. **Exit criteria:** Gate 0.
16. **Compute cost:** none.
17. **Commit:** `docs/`, tag `spec-v1`.

---

## PHASE 1: Repository and configuration

**Status (v2.0): MVP — v2 Phase 1.**

1. **Objective:** A tested skeleton with a validated config system, seeding, run metadata, atomic I/O and structured logging.
2. **Why:** Reproducibility and resumability are cheap to build now and very expensive to retrofit.
3. **Tasks:** See "First implementation step" at the end.
4. **Files to create:** `pyproject.toml`, `Makefile`, `configs/base.yaml`, `core/{config,seeding,runmeta,jsonl,paths,atomic}.py`, `cli.py`, tests.
5. **Files to modify:** none.
6. **Classes/functions:** `RootConfig`, `load_config`, `config_hash`, `SeedManager`, `collect_run_metadata`, `JsonlWriter`, `RunPaths`, `atomic_write_*`.
7. **Inputs:** `spec-v1`.
8. **Outputs:** an installable package, a run directory with a frozen config and metadata.
9. **Config parameters:** the full set from E.7 plus budget, data and logging sections.
10. **Unit tests:** about 25, listed in the first step.
11. **Integration tests:** `init-run` end to end.
12. **Expected result:** all tests pass on CPU in under a minute.
13. **Failure conditions:** a scientific parameter with a Python default; non-canonical hashing.
14. **Debugging strategy:** print the canonical JSON that is hashed; diff two configs.
15. **Exit criteria:** the first-step success criteria.
16. **Compute cost:** none.
17. **Commit:** everything except `uv.lock` caches; tag `phase-1`.

---

## PHASE 2: Environment abstraction and the first three environments

**Status (v2.0): MVP — v2 Phase 2. The remaining five environments of v1 Phase 11 are Stretch (S-11).**

1. **Objective:** The frozen interface, the split machinery, and three working environments: GSM8K, Countdown, noisy-reward.
2. **Why:** Everything downstream consumes `Prompt`, `Verdict` and split manifests. Leaks are impossible to repair later.
3. **Tasks:** Add `Prompt`, `Verdict`, `RolloutGroup` to `core/types.py`. Write `envs/base.py` and the registry. Implement `SplitManager` and `scripts/build_splits.py`. Implement GSM8K, Countdown (safe AST evaluator) and the noisy wrapper. Write the sealed-test guard skeleton. Write the contract suite and the per-environment profile script.
4. **Files to create:** `envs/{base,registry,splits,gsm8k,countdown,noisy}.py`, `evaluation/guard.py`, `scripts/{download_data,build_splits}.py`, `configs/env/*.yaml`, `tests/contract/test_env_contract.py`, `tests/unit/{test_splits,test_guard}.py`, `data/manifests/*`.
5. **Files to modify:** `core/types.py`, `configs/base.yaml` (data section), `tests/architecture/test_import_rules.py`.
6. **Classes/functions:** `Environment`, `EnvSpec`, `EnvRegistry`, `SplitManager`, `SealedTest`, `Gsm8kEnv`, `CountdownEnv`, `NoisyRewardEnv`, `safe_arith_eval`.
7. **Inputs:** raw datasets, split salt, seeds.
8. **Outputs:** processed JSONL, committed manifests, `reports/env_profiles/`.
9. **Config parameters:** `data.root`, split sizes (calib, dev), `data.split_salt`, Countdown number count and target range, noisy mode and q, per-environment token limits, verifier timeout.
10. **Unit tests:** Part M items 1, 2 and 14 for these three environments.
11. **Integration tests:** `build_splits` run twice gives identical manifest hashes; the contract suite passes for all three environments.
12. **Expected result:** gold answers verify at ≥ 99%; garbage fails at ≥ 99%; noisy success rate within 0.02 of design.
13. **Failure conditions:** answer-parsing mismatches; any split overlap; non-reproducible manifests.
14. **Debugging strategy:** print the first 20 gold-answer failures; compare manifest hashes across machines; run the MinHash report.
15. **Exit criteria:** Gate 1 (three environments).
16. **Compute cost:** CPU minutes plus dataset downloads.
17. **Commit:** code and manifests. Never raw data or the sealed test.

---

## PHASE 3: Curator simulator

**Status (v2.0): MVP — v2 Phase 3. Buffer phase: this gates all scheduler work.**

1. **Objective:** A synthetic world with known dynamics, oracles, scenarios and a harness.
2. **Why:** It is the only place the true optimum is known (Rule 3), and the scheduler is developed against it.
3. **Tasks:** Implement the world (F.2), the scenario files (F.3), the DP, static and myopic oracles (F.4), the harness and oracle-gap metric (F.5), and the replay skeleton. Add `BaseScheduler` with Uniform and Random stubs to validate the harness.
4. **Files to create:** `simulator/{world,scenarios,oracle,harness,replay}.py`, `configs/sim/*.yaml`, `experiments/run_sim.py`, tests.
5. **Files to modify:** `core/types.py` (`RoundObservation`, `CalibrationObservation`), `scheduler/base.py`.
6. **Classes/functions:** `SimWorld`, `SimEnvParams`, `ScenarioSpec`, `DPOracle`, `StaticOracle`, `MyopicOracle`, `run_episode`.
7. **Inputs:** scenario YAMLs, seeds.
8. **Outputs:** observation streams in the real schema, oracle values, per-run logs.
9. **Config parameters:** per-scenario environment parameters and their distributions, R, M, G, n_b, budget, concentration κ_b.
10. **Unit tests:** p ∈ (0,1); learnability peaks at p = 0.5; a zero-transfer environment never changes other skills; the noisy environment never changes skill; per-round cost equals M·Σ w_i c_i; the DP value is at least the value of every fixed mixture on the same grid; determinism by seed.
11. **Integration tests:** every scenario runs end to end with Uniform and Random.
12. **Expected result:** oracle beats uniform by more than 10% oracle gap in every scenario, otherwise the scenario cannot discriminate and is redesigned.
13. **Failure conditions:** scenarios where uniform ≈ oracle; DP too slow; hidden variables leaking into observations.
14. **Debugging strategy:** plot skill trajectories under oracle versus uniform; inspect one scenario by hand.
15. **Exit criteria:** tests pass; every scenario has oracle gap > 10%.
16. **Compute cost:** CPU minutes.
17. **Commit:** simulator code and scenario files.

---

## PHASE 4: Signal Engine and status classifier

**Status (v2.0): MVP — v2 Phase 5 (signals).**

1. **Objective:** Pass rate, LP (selected estimator), signal richness, proxy reward and status classification, with thresholds chosen on the simulator.
2. **Why:** These signals are the scheduler's only input; their noise properties decide whether UCB works.
3. **Tasks:** Implement E.2–E.5. Run the LP estimator study and the threshold tuning on tuning seeds. Record both outcomes in `DECISIONS.md`.
4. **Files to create:** `signals/{passrate,progress,richness,status,proxy,engine}.py`, `experiments/analysis/{lp_estimator_study,tune_status_thresholds}.py`, tests.
5. **Files to modify:** `configs/base.yaml` (signals, richness, status sections).
6. **Classes/functions:** `PassRateTracker`, `LPEstimator` (A, B, C), `RichnessEstimator`, `StatusClassifier`, `ProxyReward`, `SignalEngine`.
7. **Inputs:** `RoundObservation` streams from the simulator.
8. **Outputs:** per-environment `SignalVector` and `EnvStatus` each round; chosen estimator and thresholds.
9. **Config parameters:** λ, λ_f, λ_s, W, `lp_use`, richness mode and band, `n_min`, `r_min`, `p_sat`, `p_hard`, `sr_hard`, `z_up`, `z_neg`, hysteresis gaps, `D_min`, h.
10. **Unit tests:** Part M items 3, 4, 9 plus the property tests.
11. **Integration tests:** the engine on every simulator scenario.
12. **Expected result:** status macro-F1 ≥ 0.85 against true status, flip_rate ≤ 5 per 100 rounds, LP false-positive rate ≤ 5% on evaluation seeds.
13. **Failure conditions:** flapping; LP significant on noisy environments; richness degenerate at small G.
14. **Debugging strategy:** plot p̂, interval, LP and z over rounds beside true skill for failing cases.
15. **Exit criteria:** the Expected-result numbers hold on seeds disjoint from tuning.
16. **Compute cost:** CPU hours for the studies.
17. **Commit:** signals code, study scripts, `DECISIONS.md` entries.

---

## PHASE 5: Discounted-UCB scheduler (Curator v0) and simple baselines

**Status (v2.0): MVP — v2 Phases 5–6. The SW-UCB/EXP3/DTS variants in the last task bullet are Stretch (S-12).**

1. **Objective:** A correct discounted-UCB Curator with the Part E.6 mixture map, plus Uniform, Static, Learning-progress and Standard-UCB baselines, all in the simulator.
2. **Why:** Correctness before extras (Rule 11). Gate 2 depends on it.
3. **Tasks:** Implement `BaseScheduler`, the mixture map with floors, quotas and status constraints, D-UCB, the `Curator` facade (`update_observation`, `compute_scores`, `select_mixture`, `get_state`, `save_checkpoint`, `load_checkpoint`; `update_calibration` and `get_roi` as stubs), and the baselines. Tune γ, κ, τ, ε on tuning seeds. Run the scenario suite.
4. **Files to create:** `scheduler/{base,mixture,ducb,curator}.py`, `scheduler/baselines/{uniform,static,lp,ucb,random}.py`, `configs/scheduler/*.yaml`, `experiments/sweeps/tune_ducb_sim.py`, tests.
5. **Files to modify:** `simulator/harness.py` (scheduler factory).
6. **Classes/functions:** `BaseScheduler`, `DiscountedUCB`, `mixture_from_scores`, `apply_status_constraints`, `Curator`, `LPCurriculum`, `StandardUCB`.
7. **Inputs:** `SignalVector`, `EnvStatus`, costs.
8. **Outputs:** `MixtureDecision` per round; scheduler checkpoint.
9. **Config parameters:** γ, κ, τ, ε, `score_norm`, `pull_unit`, `warmup_rounds`, `cost_exponent`, `status_control`, caps and multipliers.
10. **Unit tests:** Part M items 6, 7, 8, 13.
11. **Integration tests:** Part M integration test 1.
12. **Expected result:** Gate 2 numbers.
13. **Failure conditions:** weights collapse onto one environment; oscillation; junk environment keeps weight; overflow.
14. **Debugging strategy:** plot UCB components (μ̂, bonus) per environment next to true marginal gain.
15. **Exit criteria:** Gate 2.
16. **Compute cost:** CPU hours (50 seeds × scenarios × methods).
17. **Commit:** scheduler code, tuned configs, sweep results.

---

## PHASE 6: Cost measurement

**Status (v2.0): MVP — v2 Phase 7a (cost measurement).**

1. **Objective:** A verified Cost Meter, ledger and attribution model with a calibration run on real hardware.
2. **Why:** Cost is the project's central variable and the easiest to get quietly wrong.
3. **Tasks:** Implement span timers with an injectable clock, the ledger with the category and stop rules (H.6), GPU sampling, and the attribution model. Run `cost_calibration.py` on Qwen2.5-0.5B: generation and verification per environment (200 prompts × 3 repeats), and update-cost micro-benchmarks on synthetic batches. Write `COST_MODEL.md`.
4. **Files to create:** `cost/{meter,ledger,gpu_stats,model}.py`, `scripts/cost_calibration.py`, `configs/trainer/qwen05b_lora.yaml` (initial), tests.
5. **Files to modify:** `core/types.py` (cost records), `docs/COST_MODEL.md`.
6. **Classes/functions:** `CostMeter.span()`, `BudgetLedger`, `GpuSampler`, `AttributionModel.fit/attribute`.
7. **Inputs:** base model, environments from Phase 2.
8. **Outputs:** per-environment unit costs with CV, fitted attribution coefficients, a cost report.
9. **Config parameters:** `cost.usd_per_gpu_hour`, `cost.n_gpus`, `cost.count_cpu_verifier_as_gpu_time`, warm-up steps, drift-monitor period.
10. **Unit tests:** Part M items 5 and 10.
11. **Integration tests:** reconciliation on a short real micro-run.
12. **Expected result:** per-environment CV < 10%; cost ordering as expected; unattributed < 3%.
13. **Failure conditions:** unattributed time; high variance; code or tool use cheaper than GSM8K.
14. **Debugging strategy:** log every span with its environment and token counts; compare to `nvidia-smi` utilisation.
15. **Exit criteria:** Gate 5 prerequisites (verification inside the trainer closes Gate 5 in Phase 8).
16. **Compute cost:** about 2–4 GPU-hours.
17. **Commit:** code, cost report, no model weights.

---

## PHASE 7: Static-mixture GRPO baseline

**Status (v2.0): MVP — v2 Phase 7b (GRPO baseline).**

1. **Objective:** A working, resumable Qwen2.5-0.5B LoRA GRPO run on a fixed uniform mixture of the three environments, independent of Curator.
2. **Why:** If plain GRPO does not train, nothing built on it can be trusted (Gate 3).
3. **Tasks:** Run the TRL spike (G.2) and record `TRL_SPIKE.md`. Pin versions. Write `TrainerAdapter`, `RewardWrapper`, `RolloutLedger`, the checkpoint code and a fixed `MixtureState`. Train uniform for 3 seeds. Set R, P, G, learning rate, KL coefficient and K from the pilot.
4. **Files to create:** `trainer/{trl_adapter,reward_wrapper,sampler,round_controller,checkpoint}.py`, `experiments/run_pilot.py`, `tests/integration/test_mock_trainer.py`, `tests/gpu/test_tiny_grpo.py`, `docs/TRL_SPIKE.md`.
5. **Files to modify:** `configs/trainer/qwen05b_lora.yaml`, `pyproject.toml` (train extra pins).
6. **Classes/functions:** `TrainerAdapter`, `TrlGrpoAdapter`, `RewardWrapper`, `RolloutLedger`, `save/load_training_state`.
7. **Inputs:** environments, cost model.
8. **Outputs:** training curves, ledger, per-environment pass rates, resumable checkpoints.
9. **Config parameters:** LoRA rank and targets, G, P, R, completion lengths, learning rate, warmup, KL coefficient, precision, generation backend.
10. **Unit tests:** reward wrapper routing; group-integrity assertion; checkpoint round trip.
11. **Integration tests:** mock trainer; tiny-model GRPO smoke; kill-and-resume.
12. **Expected result:** GSM8K train pass rate and calibration score rise beyond noise.
13. **Failure conditions:** no learning; memory errors; non-resumable runs; version drift in TRL internals.
14. **Debugging strategy:** compare to a reference TRL GRPO example on one environment; check reward-function outputs by hand on 20 samples; audit 50 completions per environment.
15. **Exit criteria:** Gate 3.
16. **Compute cost:** about 5–10 GPU-hours.
17. **Commit:** trainer code, configs, spike notes.

---

## PHASE 8: Curator + GRPO integration

**Status (v2.0): MVP — v2 Phase 7c (integration).**

1. **Objective:** Curator controls the environment mixture inside the real trainer without changing GRPO.
2. **Why:** This is the main engineering risk, so it is validated separately (Gate 4).
3. **Tasks:** Implement `CuratorPromptStream`, `MixtureState` updates in `CuratorCallback`, observation building, budget stop and atomic checkpoints. Run the passthrough equivalence test. Run Curator v0 (proxy-only). Validate cost attribution against the per-environment `rollout_func` cross-check.
4. **Files to create:** `trainer/callback.py`, `trainer/prompt_stream.py`, `tests/integration/test_passthrough.py`, `tests/gpu/test_attribution.py`.
5. **Files to modify:** `trainer/trl_adapter.py`, `trainer/round_controller.py`.
6. **Classes/functions:** `CuratorCallback`, `CuratorPromptStream`, `MixtureState`, `build_round_observation`.
7. **Inputs:** Curator v0, Phase 7 trainer.
8. **Outputs:** a real run with logged weights, signals, costs.
9. **Config parameters:** R, K, `budget.total_usd`, `stop_on_target` (false), prefetch settings.
10. **Unit tests:** quota rounding; stream respects weights; ledger-to-observation mapping.
11. **Integration tests:** passthrough equivalence; realised-proportion test; resume.
12. **Expected result:** the Gate 4 and Gate 5 criteria.
13. **Failure conditions:** weight-update lag; group integrity broken; hidden changes to optimiser behaviour.
14. **Debugging strategy:** run with fixed uniform weights and diff against Phase 7 logs step by step.
15. **Exit criteria:** Gates 4 and 5.
16. **Compute cost:** about 10 GPU-hours.
17. **Commit:** integration code and test results.

---

## PHASE 9: Held-out calibration

**Status (v2.0): MVP — v2 Phase 8a. Buffer phase: the main novelty claim.**

1. **Objective:** The calibration benchmark, the cost-charged evaluator, α/β fitting, credit rules and S5 detection.
2. **Why:** It is the second speed of the reward and the main novelty claim.
3. **Tasks:** Build the per-domain calibration slices. Implement the evaluator and ΔS with paired-bootstrap standard errors. Implement the estimator comparison in the simulator, then the chosen α/β fit, C1, C2 and C2d credit rules, and S5. Sweep K.
4. **Files to create:** `calibration/{benchmark,evaluator,regression,credit}.py`, `experiments/analysis/ab_estimator_study.py`, tests.
5. **Files to modify:** `scheduler/curator.py` (`update_calibration`), `signals/status.py` (S5 input), `trainer/callback.py`.
6. **Classes/functions:** `CalibrationBenchmark`, `CalibrationEvaluator`, `AlphaBetaFitter`, `ShareCredit`, `RegressionCredit`.
7. **Inputs:** checkpoints, window exposures, signals.
8. **Outputs:** ΔS series, fitted α and β with diagnostics, per-environment gain estimates.
9. **Config parameters:** slice sizes, K, `calib.target`, ν grid, `k_min`, trust region, `z_mis`, q.
10. **Unit tests:** Part M items 11 and 14.
11. **Integration tests:** the simulator with planted α and β; a Tier 1 run with calibration.
12. **Expected result:** the Gate 6 criteria.
13. **Failure conditions:** calibration set leaking into training; unstable α and β; calibration cost above 10% of budget.
14. **Debugging strategy:** plot proxy-implied against observed gain per window and environment.
15. **Exit criteria:** Gate 6.
16. **Compute cost:** about 10–15 GPU-hours.
17. **Commit:** calibration code, study outputs.

---

## PHASE 10: ROI engine

**Status (v2.0): MVP — v2 Phase 8b (ROI). On completion: Milestone M-1, MVP functionally complete.**

1. **Objective:** Per-environment ROI with intervals, reconciliation and a leaderboard.
2. **Why:** The leaderboard is a headline output and must be checked against known truth.
3. **Tasks:** Implement the engine (I.6), the bootstrap, the reconciliation check and the leaderboard export (CSV, JSON, simple HTML). Validate on the simulator against true marginal ROI.
4. **Files to create:** `roi/{engine,bootstrap,leaderboard}.py`, tests.
5. **Files to modify:** `scheduler/curator.py` (`get_roi`), `viz/` (leaderboard figure).
6. **Classes/functions:** `RoiEngine`, `bootstrap_roi`, `Leaderboard`.
7. **Inputs:** credited gains, cost ledger.
8. **Outputs:** `roi_leaderboard.{csv,json,html}`.
9. **Config parameters:** bootstrap draws, interval level, minimum windows.
10. **Unit tests:** Part M item 12.
11. **Integration tests:** the simulator, with Spearman against true ROI.
12. **Expected result:** Gate 7 numbers.
13. **Failure conditions:** reconciliation residual large; intervals too narrow in the simulator.
14. **Debugging strategy:** compare per-window credits against planted truth.
15. **Exit criteria:** Gate 7.
16. **Compute cost:** CPU.
17. **Commit:** ROI code.

---

## PHASE 11: Remaining environments and all baselines on GRPO

**Status (v2.0): v2 Phase 9 for the MVP subset — Tier-1 runs of the MVP methods only, plus the priority ablations listed in Phase 12. The five new environments are Stretch (S-11); the SEC baseline is Stretch (S-13).**

1. **Objective:** Add MATH, MBPP, Knights-and-Knaves, tool-use and too-hard, and run every baseline reproducibly.
2. **Why:** The proposal's full portfolio and the fair-comparison evidence.
3. **Tasks:** Implement the five environments and the code sandbox; build the BFCL sealed slice. Add baselines in the J.1 order. Verify matched budgets and reproducibility.
4. **Files to create:** `envs/{math_env,mbpp,knights_knaves,toolcall,toohard,sandbox}.py`, `configs/env/*.yaml`, `configs/scheduler/*.yaml`, tests.
5. **Files to modify:** `data/manifests/`, `evaluation/guard.py`, `scheduler/baselines/`.
6. **Classes/functions:** the environment classes; `SecStyleBandit` (optional).
7. **Inputs:** datasets, Phase 2 machinery.
8. **Outputs:** Tier 1 results for every baseline across seeds.
9. **Config parameters:** per-environment parameters; per-baseline hyperparameters.
10. **Unit tests:** the contract suite for each new environment.
11. **Integration tests:** one short run per baseline; budget-matching check.
12. **Expected result:** Gate 1B and Gate 8.
13. **Failure conditions:** sandbox escapes or hangs; hacking of verifiers; baselines not budget-matched.
14. **Debugging strategy:** manual audit of 50 completions per environment; replay a baseline twice and compare.
15. **Exit criteria:** Gates 1B and 8.
16. **Compute cost:** about 40–60 GPU-hours.
17. **Commit:** code and manifests.

---

## PHASE 12: Ablations

**Status (v2.0): v2 Phase 9 for the priority ablations; the full grid is Stretch (S-14). In the MVP, run Tier 0 for all ablations (free) and Tier 1 only for `cost_exponent: 0` and `calib.enabled: false` if the schedule allows; C1-vs-C2 is free by re-analysis of logged windows.**

1. **Objective:** Test the central hypotheses first, then the sensitivity analyses.
2. **Why:** Shows which parts of Curator actually matter (H1–H7).
3. **Tasks:** Run Tier 0 for every ablation, Tier 1R replays for scheduler-logic ones, then Tier 1 for the priority set.
4. **Files to create:** `experiments/sweeps/ablations_*.yaml`, `experiments/analysis/ablation_tables.py`.
5. **Files to modify:** none.
6. **Classes/functions:** config flags only.
7. **Inputs:** the Curator code with flags.
8. **Outputs:** ablation tables with intervals.
9. **Config parameters:** see the table below.
10. **Unit tests:** each flag changes behaviour in the intended direction in the simulator.
11. **Integration tests:** every flag combination in the sweep config loads and runs one round.
12. **Expected result:** a ranked list of component contributions.
13. **Failure conditions:** an ablation that is not isolated (two flags change at once).
14. **Debugging strategy:** diff config hashes between the ablation and the full method.
15. **Exit criteria:** priority ablations complete with 3 seeds at Tier 1.
16. **Compute cost:** about 27 GPU-hours at Tier 1 (planning assumption).
17. **Commit:** sweep configs, tables.

| Priority | Your letter | Ablation | Flag |
|---|---|---|---|
| 1 | A | No cost normalisation | `cost_exponent: 0` |
| 1 | B | No benchmark calibration | `calib.enabled: false` |
| 1 | F | Standard UCB instead of discounted | `gamma: 1.0` (plus no cost/calibration for the baseline) |
| 1 | new | Share credit (C1) instead of regression credit (C2) | `credit.rule` |
| 2 | E | No discounting | `gamma: 1.0` |
| 2 | G | No exploration floor | `epsilon: 0` |
| 3 | C, D | No signal richness; no learning progress | `proxy.beta: 0`; `proxy.alpha: 0` |
| 3 | I, J, K, L | Sensitivity to K, τ, ε, γ | grids |
| n/a | H | "No ROI validation" is an evaluation step, not a training change. Report ROI with and without LOO and the effect of LOO subset size | none |

---

## PHASE 13: Leave-one-out validation

**Status (v2.0): MVP — v2 Phase 10, at Tier 1 only. The Tier-2 LOO subset is Stretch (with S-16).**

1. **Objective:** The LOO pipeline and its results, first in the simulator, then Tier 1, then a Tier 2 subset.
2. **Why:** It is the project's attribution evidence.
3. **Tasks:** Implement the pipeline (Part K), run it in the simulator against exact counterfactuals, then Tier 1, then Tier 2 on the pre-registered subset.
4. **Files to create:** `evaluation/loo.py`, `experiments/run_loo.py`, tests.
5. **Files to modify:** `evaluation/stats.py`.
6. **Classes/functions:** `LeaveOneOutRunner`, `rank_agreement`, `bootstrap_over_seeds`.
7. **Inputs:** full-run logs, removed-environment configs.
8. **Outputs:** contribution tables, rank agreement with intervals.
9. **Config parameters:** subset, seeds, base variant (L-C or L-U), evaluation split `dev`.
10. **Unit tests:** Part M item 15.
11. **Integration tests:** simulator LOO against the true counterfactual.
12. **Expected result:** the Gate 9 criteria.
13. **Failure conditions:** budget not matched; subset chosen after seeing results; circular evaluation on `calib`.
14. **Debugging strategy:** verify realised compute per environment in each LOO run.
15. **Exit criteria:** Gate 9.
16. **Compute cost:** about 18 GPU-hours at Tier 1 and about 40 at Tier 2 (planning assumption).
17. **Commit:** pipeline and results.

---

## PHASE 14: Three-seed main experiments

**Status (v2.0): Stretch (S-16) at Tier 2. The Tier-1 three-seed comparison (the MVP headline, v2 Phase 9) uses the v1 Phase 11 machinery.**

1. **Objective:** The final multi-seed comparison at Tier 2.
2. **Why:** It produces the headline result.
3. **Tasks:** Freeze configs, commit the analysis plan, run the matrix (Part R), monitor via logged curves (the offline dashboard; a live monitor is Stretch S-16), resume as needed.
4. **Files to create:** `docs/ANALYSIS_PLAN.md`, `experiments/run_main.py`, `reports/frozen_*.hash`.
5. **Files to modify:** none.
6. **Classes/functions:** none new.
7. **Inputs:** frozen configs.
8. **Outputs:** all run directories with logs and checkpoints.
9. **Config parameters:** frozen.
10. **Unit tests:** the full test suite must be green at the frozen commit.
11. **Integration tests:** the full-pipeline dry run on a tiny model.
12. **Expected result:** complete logs for every cell of the matrix.
13. **Failure conditions:** dirty Git tree; changed configs mid-matrix; unmatched budgets.
14. **Debugging strategy:** the budget-overshoot report per run; compare early rounds across seeds.
15. **Exit criteria:** Gate 10 before starting, then completion of all cells.
16. **Compute cost:** about 105 GPU-hours (planning assumption).
17. **Commit:** the analysis plan and frozen hashes. Logs are archived, not committed.

---

## PHASE 15: Final untouched test

**Status (v2.0): MVP — v2 Phase 10, run on the MVP's final Tier-1 checkpoints.**

1. **Objective:** One evaluation of every final checkpoint on the sealed test set.
2. **Why:** The only unbiased estimate of generalisation.
3. **Tasks:** Run the guard-protected evaluator once per final checkpoint, write the results table, and archive the access log.
4. **Files to create:** `evaluation/final_test.py`, `experiments/run_final_test.py`.
5. **Files to modify:** none.
6. **Classes/functions:** `run_final_test`.
7. **Inputs:** final checkpoints, frozen config hashes.
8. **Outputs:** `reports/final_test_results.json`, `reports/test_access_log.jsonl`.
9. **Config parameters:** none tunable.
10. **Unit tests:** the guard tests, rerun.
11. **Integration tests:** a dry run against a fake sealed set.
12. **Expected result:** exactly one access entry per checkpoint.
13. **Failure conditions:** a second access; an unfrozen config.
14. **Debugging strategy:** read the access log before and after.
15. **Exit criteria:** log verified; results written.
16. **Compute cost:** about 2 GPU-hours.
17. **Commit:** results and access log.

---

## PHASE 16: Plots, tables and research report

**Status (v2.0): MVP — v2 Phase 10. On completion: Milestone M-2, MVP paper-ready.**

1. **Objective:** Every figure and table in Part S, and the written report.
2. **Why:** Turns logs into evidence.
3. **Tasks:** Implement the plot modules, generate tables from logs only, write the report with the limitations in K.3 and T.
4. **Files to create:** `viz/*.py`, `experiments/analysis/make_tables.py`, `reports/`.
5. **Files to modify:** none.
6. **Classes/functions:** plot functions.
7. **Inputs:** logs, results.
8. **Outputs:** figures, tables, report.
9. **Config parameters:** figure styles.
10. **Unit tests:** plot functions run on fixture logs.
11. **Integration tests:** `make report` from archived logs.
12. **Expected result:** all figures regenerate from logs with one command.
13. **Failure conditions:** a number in the report that no script produces.
14. **Debugging strategy:** regenerate from scratch on a clean checkout.
15. **Exit criteria:** Part U checklist complete.
16. **Compute cost:** none.
17. **Commit:** figures, tables, report source.

---

# PART Q: Milestone gates

A gate passes only if every criterion is met. Record the evidence in `docs/GATES.md`. **Gate rows name v1.0 phases; the v2.0 equivalents are in the Part P mapping table.**

| Gate | Objective | Pass criteria |
|---|---|---|
| **0** Spec frozen + scope frozen | v2 Phase 0 done | `spec-v1` tagged; all 15 defects resolved or accepted with a reason; notation table has no duplicate symbols; split salt chosen; **A0.2 MVP/Stretch table and A.4 ownership/calendar frozen into `SPEC.md`; proposal dashboard/verl wording corrections recorded in `DECISIONS.md` and applied to the proposal document** |
| **1** Interface (first 3 environments) | Phase 2 done | Contract suite green for GSM8K, Countdown, noisy; manifests reproducible; leakage tests pass; gold ≥ 99%, garbage ≤ 1%; noisy rate within 0.02 |
| **1B** Full portfolio | Phase 11 | Same suite green for MATH, MBPP (sandbox timeout works), K&K, tool-use, too-hard (base pass rate ≤ 0.02); BFCL slice sealed |
| **2** Simulator recovery | v1 Phase 5 = v2 Phases 5–6 | **Primary criteria (all required):** (a) robust improvement over Uniform in S-A — paired mean difference over seeds with a 95% interval excluding 0; (b) no underperformance versus Uniform in S-B — the paired interval's lower bound is above −δ_gap (a small pre-registered tolerance, e.g. 1 point); (c) junk compute share ≤ 1.5× floor after round 30 in ≥ 90% of seeds; (d) status F1 ≥ 0.85 and flip_rate ≤ 5 per 100 rounds; (e) beats uniform in compute-to-target by ≥ 15% in S-C (paired, interval excludes 0). **Development target, not a pass/fail threshold:** oracle-gap closure = (S_Curator − S_uniform) / (S_oracle − S_uniform) of ≥ 75% in S-A; a result in the 60–75% range with a tight interval and seed-consistent behaviour is acceptable, provided the shortfall is analysed and documented in `docs/DECISIONS.md` before moving to v2 Phase 7 |
| **3** Static GRPO works | Phase 7 | Uniform runs reproduce a learning curve in 3 seeds: calibration score gain ≥ 2 SE and GSM8K train pass rate up ≥ 5 points; kill-and-resume works; memory within limits |
| **4** Curator does not alter GRPO | Phase 8 | **Equivalence-margin testing, not non-significance.** Before the test, register a practical equivalence margin δ per metric in `docs/EXPERIMENT_PROTOCOL.md` (e.g. paired final score |Δ| < δ_S with δ_S = 1 benchmark point; mean per-step reward |Δ| < δ_R with δ_R = 0.01). Pass criterion: |S_adapter − S_reference| < δ on each pre-chosen metric over 3 paired seeds. Distribution shape checks (two-sample KS on reward, loss, KL) are recorded as diagnostics only — a p > 0.05 result is reported but is not treated as proof of equivalence. Plus: realised shares within ±2 points; group integrity holds; update lag ≤ 1 step; overhead < 2% of round time |
| **5** Cost verified | Phase 8 | Attributed per-environment cost within 10% of exact single-environment timing; ledger sums to measured wall × GPUs within 2%; unit-cost CV < 10%; unattributed < 3%; cost ordering sane |
| **6** Calibration works | Phase 9 | Simulator: recovers planted α, β (R² ≥ 0.8 at the stated noise) and beats the naive baseline by ≥ 20% out-of-window; Tier 1: α, β ≥ 0, leave-one-window-out error below the null model, noisy environment flagged S5 within 8 windows in ≥ 2 of 3 seeds; test-access log empty; calibration cost ≤ 10% of budget |
| **7** ROI leaderboard | Phase 10 | Reconciliation residual reported (zero for C1); ROI from measured costs only; simulator Spearman vs true marginal ROI ≥ 0.8 (N = 8); junk ranked last in ≥ 90% of seeds; interval coverage ≥ 90% |
| **8** Baselines reproducible | Phase 11 | Each baseline rerun with the same seed gives final score within 1 SE and the same config hash; every run's cost within 2% of B |
| **9** LOO works | Phase 13 | Simulator LOO recovers true counterfactual contributions within ±10%; Tier 1 pipeline runs end to end on ≥ 3 environments with paired seeds and bootstrap intervals; budgets matched; evaluation on `dev` |
| **10** **Experiment Freeze Gate** (final experiment ready) | Before the expensive runs (v1 Phase 14 / the v2 Phase 9–10 comparison) | **The full experimental protocol is frozen:** environments and their dataset versions (manifest hashes), model checkpoint, prompt templates, reward definitions, compute budget B, seed list, calibration frequency K, cost definition (`COST_MODEL.md`), baseline implementations (config hashes), evaluation metrics and decoding protocol, test-set access rules, ROI calculation, LOO subset and procedure, equivalence margins δ, and `ANALYSIS_PLAN.md` committed. Configs frozen by hash; full test suite green on a clean tree; dry run on a tiny model passes; sealed-test guard tested; estimated GPU-hours ≤ 80% of available. **After this gate, any change requires a documented protocol revision in `docs/DECISIONS.md` with its justification and affected claims** |
| **M-1** MVP functionally complete (v2.0) | v1 Phases 6–10 done | Gates 3, 4, 5, 6, 7 passed; Curator controls real GRPO training; calibration and ROI leaderboard produced from a real run |
| **M-2** MVP paper-ready (v2.0) | v1 Phases 13, 15, 16 done at Tier 1 | Gate 9 passed; Tier-1 four-method comparison complete (3 paired seeds, matched budget); Tier-1 LOO done; sealed test accessed exactly once per checkpoint; Part S figures regenerate from logs; Part U checklist green. **Everything from here on is stretch** |

---

# PART R: Final experiment matrix

**v2.0 note.** In the MVP, the headline result is the **Development comparison row (Tier 1, 3–5 seeds) plus the Tier 1 ablations and Tier-1 LOO**. The **Main comparison (Tier 2)** and Tier-2 LOO rows are stretch (S-16) and run only after M-2. The protocol-freeze rules of Gate 10 apply to whichever comparison is actually run.

| Block | Tier | Methods | Seeds | Runs | Purpose |
|---|---|---|---|---|---|
| Scheduler validation | 0 | all schedulers × scenarios S-A..S-H | ≥ 50 | thousands | Gate 2, regret, status F1 |
| Ablations | 0 | all A–L | ≥ 50 | thousands | Component value |
| Trace replay | 1R | scheduler variants | ≥ 20 | many (CPU) | Scheduler logic only |
| Pilots | 1 | cost, noise, transfer, junk check on two model families | 3 | ~15 | Calibrate simulator and K |
| Development comparison | 1 | Uniform, Static, LP, UCB, Curator v0, no cost, no calibration, Full | 3–5 | 24–40 | Pre-flight for Tier 2 |
| Tier 1 ablations | 1 | priority ablations | 3 | 18 | Central hypotheses |
| **Main comparison** | **2** | **Uniform, Static, LP, UCB, Curator no cost, Curator no calibration, Full Curator** (SEC-style if feasible) | **3** | **21 (24)** | **Headline result** |
| LOO | 2 (subset) and 1 | L-C on 4 environments (including both sanity environments) | 2 at Tier 2, 3 at Tier 1 | 8 and 12 | ROI validation |
| Final test | 2 | all final checkpoints | n/a | one access each | Generalisation |

---

# PART S: Expected tables and figures

## S.1 Tables

**Main results** (mean ± std and 95% bootstrap interval over seeds; every method at the same budget B)

| Method | Final score | ΔScore | GPU-hours | GPU-$ | Score/GPU-$ | AUC score-cost |
|---|---|---|---|---|---|---|

**Environment table** (Full Curator, plus LOO)

| Environment | Compute ($) | Estimated gain | Estimated ROI [CI] | LOO contribution | Rank (ROI) | Rank (LOO) | State |
|---|---|---|---|---|---|---|---|

Further tables: ablation results; compute-to-target; scheduler overhead; calibration diagnostics (α, β, R², CV error); status-transition statistics; per-environment unit cost with CV.

## S.2 Figures (each generated from logs by one function in `viz/`)

1. Environment weight versus round (stacked area).
2. Learning progress versus round, per environment.
3. Pass rate versus round, per environment.
4. Cost per environment (unit cost and cumulative).
5. ROI leaderboard (bars with intervals).
6. Benchmark score versus GPU-dollar, all methods (headline).
7. Benchmark score versus training steps (for contrast with 6).
8. UCB score (μ̂ plus bonus) versus round.
9. Environment state transitions (timeline per environment).
10. Score-cost curves with AUC shading.
11. Curator versus each baseline (paired differences with intervals).
12. ROI ranking versus LOO contribution (scatter with rank labels).
13. Calibration: proxy-implied versus observed gain per window; α and β over time.

---

# PART T: Risks and failure modes

| Problem | Why it matters | Detection | Mitigation | Where |
|---|---|---|---|---|
| Benchmark leakage | Inflates every result | Disjointness and MinHash tests; split manifests | Hash-based splits; separate manifests; leakage tests in CI | `envs/splits.py`, `tests/unit/test_splits.py` |
| Incorrect attribution | A wrong leaderboard is the main claim | Rank agreement vs LOO; simulator truth | Regression credit C2; C1 kept as baseline only; intervals; wording "validated attribution" | `calibration/credit.py`, `roi/` |
| Reward hacking | Pass rate rises without real skill | Proxy-vs-benchmark gap; audit 50 completions per environment | Strict parsers; hidden tests for code; sandbox; S5 | `envs/*`, `signals/status.py` |
| Proxy misalignment | Richness rewards noise | Noisy environment in the portfolio; S5 | Calibration; shrinkage; signed LP | `calibration/regression.py` |
| Cost measurement bias | Corrupts ROI and rewards | Reconciliation; CV; drift monitor | Spans with sync; attribution validated vs exact timing | `cost/` |
| Environment interaction | Violates additive credit | Simulator S-D; LOO non-additivity checks | Report; C2d; LOO limitations | `simulator/`, `evaluation/loo.py` |
| Difficulty confounded with cost | Hard environments look both costly and low-gain | Log both; partial analysis | Report gain and cost separately; `cost_exponent` ablation | `roi/`, analysis |
| Scheduler overhead | Could erase the saving | Overhead metric | Charge overhead; keep scheduler O(N); K rule | `cost/`, `trainer/callback.py` |
| Unfair baselines | Invalidates the claim | Fairness checklist; config diff | Same budget, seeds, optimiser; equal tuning | `docs/EXPERIMENT_PROTOCOL.md` |
| Different numbers of rollouts | Cheaper environments buy more rollouts | Report covariates | This is the point of cost awareness; report steps, rollouts, tokens | `evaluation/metrics.py` |
| Different token lengths | Drive cost and may drive reward | Token logs | Report; cap lengths identically | `envs/`, `cost/` |
| Seed variance | 3 seeds is little power | Std and intervals; Tier 0 power | No claims on overlapping intervals; simulator for power | `evaluation/stats.py` |
| Calibration overfitting | Fitting α, β to noise | Leave-one-window-out CV; null model | Ridge to prior; non-negativity; trust region; fallback | `calibration/regression.py` |
| Test-set contamination | Invalidates final claims | Access log; guard | Sealed directory; env var; frozen config hash; single access | `evaluation/guard.py` |
| Unstable UCB estimates | Early rounds dominated by noise | Plot μ̂ and bonus | Warm-up; reward mapping; z-scoring; Beta smoothing | `scheduler/ducb.py` |
| Insufficient exploration | Misses a late-blooming environment | S-F, S-H scenarios | Floor ε; S1 quotas; revival probes | `scheduler/mixture.py` |
| Catastrophic starvation | An environment is dropped permanently | Minimum-weight monitor | Floor; hard cap never below ε/N; alerts | `scheduler/mixture.py` |
| Spurious rewards on Qwen | A random-reward environment may still raise scores | Junk check on a second model family | Treat the noisy environment as a probe, not a known-useless arm | Phase 6/9 pilots |
| Pretraining contamination (GSM8K, MATH) | Gains may reflect memorisation | Report procedural environments separately | Procedural environments as clean signal | reports |
| Learning-rate schedule tied to steps | Unequal training across methods | Config diff | Constant LR after warmup | `trainer/` configs |
| TRL version drift | Hooks break silently | Adapter tests | Pin versions; one adapter file; version-change test | `trainer/trl_adapter.py` |
| Early-stop on target | Breaks matched compute | Config check | `stop_on_target` false in all matched runs | `trainer/callback.py` |

---

# PART U: Final checklist before claiming results

- [ ] `spec-v1` unchanged since Gate 0, or every change is logged in `DECISIONS.md`.
- [ ] All gates 0–10 (and 1B if stretch S-11 ran) passed with evidence in `docs/GATES.md`.
- [ ] MVP boundary respected: the claims made rest only on MVP (or documented stretch) evidence — nothing half-run is presented.
- [ ] Split manifests committed; leakage tests green on the final commit.
- [ ] Sealed test accessed exactly once per final checkpoint; access log archived.
- [ ] Every method used the same budget B under the ledger rule; overshoot reported.
- [ ] Calibration and overhead charged to Curator; reporting evaluations identical across methods.
- [ ] Constant learning rate after warmup in every run.
- [ ] At least 3 seeds per compared cell (Tier 1 in the MVP, Tier 2 in stretch); intervals reported; no claims on overlapping intervals.
- [ ] Hyperparameters tuned only on simulator tuning seeds and `dev`; frozen by hash.
- [ ] Cost attribution validated (Gate 5) and the cost model described in the report.
- [ ] ROI described as estimated marginal contribution, with LOO agreement and its limitations.
- [ ] Noisy-reward environment result reported on two model families.
- [ ] Procedural and public-benchmark environments reported separately.
- [ ] Crashed and diverged runs reported.
- [ ] Every number in the report regenerates from logs with one command.
- [ ] Clean-tree commit hash, lock file and hardware log archived.

---

# FIRST IMPLEMENTATION STEP: PHASE 1 (repository and configuration)

Gate 0 sign-off (`spec-v1`) comes first, and the Phase 1 scaffolding needs only the notation table and the hyperparameter list from this roadmap. You can draft `SPEC.md` while Phase 1 runs.

## Exact files

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package `curator-rl` (import `curator_rl`), Python ≥ 3.11, `src/` layout. Core dependencies: numpy, scipy, pandas, pyarrow, pydantic ≥ 2, pyyaml. Extras: `dev` (pytest, pytest-cov, hypothesis, ruff), `train` (empty for now, pinned in Phase 7), `sim`. Ruff and pytest settings. Marker `gpu` |
| `Makefile` | Targets `install`, `test`, `lint`, `smoke` |
| `.github/workflows/ci.yml` | GitHub Actions: `make lint && make test` on CPU only (ubuntu-latest, Python 3.11+3.12). Cheap guard for every push; satisfies the CPU-only CI invariant of O.1 |
| `.gitignore`, `.pre-commit-config.yaml`, `README.md` | Ignore `data/raw`, `data/processed`, `data/test_sealed`, `checkpoints`, `logs`, `runs`. Ruff pre-commit hook. README with the Phase 1 commands and the O.1 platform matrix (develop on Windows/WSL2, GPU runs only on Linux) |
| `configs/base.yaml` | Every scientific hyperparameter from E.7 with its documented starting value, plus budget, cost, data and logging sections |
| `configs/experiment/smoke.yaml` | A minimal experiment that inherits from `base.yaml` |
| `docs/SPEC.md`, `docs/DECISIONS.md` | Skeletons: notation table (E.1), decision-log template |
| `src/curator_rl/__init__.py` | Version string |
| `src/curator_rl/core/config.py` | Pydantic models and the loader |
| `src/curator_rl/core/seeding.py` | Named, order-independent random streams |
| `src/curator_rl/core/runmeta.py` | Git, platform, package and hardware metadata; run IDs |
| `src/curator_rl/core/jsonl.py` | Append-only JSONL writer and tolerant reader |
| `src/curator_rl/core/atomic.py` | Atomic text and JSON writes |
| `src/curator_rl/core/paths.py` | Run directory layout |
| `src/curator_rl/cli.py` | Commands `init-run`, `show-config`, `env-info` |
| `src/curator_rl/{envs,signals,scheduler,cost,calibration,roi,trainer,simulator,evaluation,viz}/__init__.py` | Empty packages, so the import-rule test has real targets |
| `tests/conftest.py`, `tests/unit/test_{config,seeding,runmeta,jsonl,atomic,paths}.py`, `tests/architecture/test_import_rules.py` | The tests below |

## Exact interfaces

```python
# core/config.py
class RootConfig(BaseModel):          # extra="forbid" on every model
    experiment: ExperimentCfg         # name, tier in {0, "1R", 1, 2}, method, seed >= 0
    budget: BudgetCfg                 # total_usd > 0, usd_per_gpu_hour > 0, n_gpus >= 1
    scheduler: SchedulerCfg           # gamma in (0,1], exploration_coef >= 0, tau > 0, epsilon in [0,1),
                                      # warmup_rounds >= 0, score_norm, cost_exponent >= 0, status_control
    signals: SignalsCfg               # lambda, lambda_fast, lambda_slow, lp_method, lp_use, richness mode/band, thresholds
    calib: CalibCfg                   # interval_rounds >= 1, target, k_min, trust_region, z_mis
    cost: CostCfg                     # count_cpu_verifier_as_gpu_time, warmup_steps
    data: DataCfg                     # root, split_salt, calib_size, dev_size
    paths: PathsCfg                  # runs_root
    logging: LoggingCfg               # fsync, schema_version

def load_config(path: Path, overrides: Sequence[str] = ()) -> RootConfig   # "a.b=c" overrides, YAML-typed
def config_hash(cfg: RootConfig) -> str                                    # SHA-256 of canonical JSON (sorted keys)
def dump_config(cfg: RootConfig, path: Path) -> None
```

Scientific hyperparameters (γ, κ, τ, ε, K, α, β, thresholds) have **no Python defaults**. A config missing one fails validation (Rule 5). Unknown keys fail validation.

```python
# core/seeding.py
class SeedManager:
    def __init__(self, master_seed: int) -> None
    def rng(self, name: str) -> np.random.Generator     # depends on (master_seed, name) only
    def int_seed(self, name: str, bits: int = 32) -> int

# core/runmeta.py
def collect_run_metadata(repo_root: Path) -> dict       # git_sha, git_dirty, diff_hash, python, platform,
                                                        # packages, gpu (nvidia-smi or None), cpu_count, timestamp_utc
def make_run_id(cfg: RootConfig, meta: dict, date: str | None = None) -> str   # {date}_{tier}_{method}_s{seed}_{git7}_{cfg6}

# core/jsonl.py
class JsonlWriter:                                     # context manager
    def __init__(self, path: Path, schema_version: int, fsync: bool = False) -> None
    def write(self, record: dict) -> None               # adds "_schema" and a monotonically increasing "_seq"
def read_jsonl(path: Path, skip_corrupt_tail: bool = True) -> Iterator[dict]

# core/atomic.py
def atomic_write_text(path: Path, text: str) -> None   # temp file in same directory, fsync, os.replace
def atomic_write_json(path: Path, obj: Any) -> None

# core/paths.py
class RunPaths:
    def __init__(self, runs_root: Path, run_id: str) -> None
    def create(self) -> None                            # idempotent: run_dir, logs/, checkpoints/, reports/
```

`init-run` behaviour: create the run directory, write the frozen `config.yaml` and `metadata.json` atomically, and print one JSON line `{"run_id": ..., "run_dir": ..., "config_hash": ...}`. If the directory already exists with the **same** config hash it is reused; with a **different** hash it exits with an error.

## Exact tests

| Test | Assertion |
|---|---|
| `test_config_loads_base_yaml` | `load_config` returns a `RootConfig` with every section populated |
| `test_missing_scientific_param_raises` | Removing `scheduler.gamma` (or any listed parameter) raises `ValidationError` |
| `test_unknown_key_raises` | A typo key raises |
| `test_invalid_ranges_raise` | γ = 0, ε = 1.0, K = 0, τ ≤ 0, total_usd ≤ 0 each raise |
| `test_override_syntax` | `scheduler.tau=0.5` sets a float; a bad path raises |
| `test_config_hash_stable_and_sensitive` | Same config gives the same 64-hex hash regardless of key order; changing any hyperparameter changes it |
| `test_spec_lint_keys_exist` | Every hyperparameter name in the SPEC notation table appears in `base.yaml` |
| `test_seed_streams_reproducible_and_independent` | Same (master, name) gives identical draws; different names are uncorrelated |
| `test_seed_order_independence` | Requesting stream `a` then `b` equals requesting `b` then `a` |
| `test_run_metadata_required_keys` | All keys present; `git_dirty` is a bool; works outside a Git repo by returning `None` for Git fields |
| `test_run_id_format` | Matches the pattern and embeds config-hash and git prefixes |
| `test_jsonl_roundtrip_and_append` | Written records read back equal; reopen appends; `_seq` continues |
| `test_jsonl_truncated_tail_skipped` | A file cut mid-line yields all complete records, and the reader reports the skip |
| `test_atomic_write_no_partial_file` | An exception injected during write leaves the old file intact and no temp file behind |
| `test_runpaths_create_idempotent` | Calling `create()` twice is safe |
| `test_init_run_idempotent_and_conflict` | Same config twice reuses the directory; a changed config errors |
| `test_import_rules` | Scanning `src/` with `ast`: L1 packages do not import torch, transformers, trl, datasets, `envs`, `trainer` or `evaluation.final_test`; only `trainer/trl_adapter.py` may import trl. Passes trivially now, and it is the guard for every later phase |

## Commands to run

```bash
uv sync --extra dev                  # or: pip install -e ".[dev]"
make lint                            # ruff check
make test                            # pytest -q tests --cov=curator_rl.core
python -m curator_rl.cli init-run --config configs/experiment/smoke.yaml --set experiment.seed=0
```

## Expected output

- `make lint`: no findings.
- `make test`: about 25 tests collected and passed, in a few seconds, with `core` coverage at or above 90%.
- `init-run` prints one JSON line such as:

```
{"run_id": "20261003_t0_smoke_s0_ab12cd3_9f31e2", "run_dir": "runs/20261003_t0_smoke_s0_ab12cd3_9f31e2", "config_hash": "9f31e2..."}
```

- The run directory contains `config.yaml`, `metadata.json`, `logs/`, `checkpoints/`, `reports/`.
- Running the same `init-run` command again prints the same line and changes nothing.

## Success criteria

1. All tests pass, including the import-rule test, on a clean checkout on CPU only.
2. `ruff check` is clean.
3. Removing any scientific hyperparameter from `base.yaml` makes `load_config` fail.
4. The same config and seed give an identical `config_hash` on two machines.
5. `init-run` is idempotent for identical configs and refuses conflicting ones.
6. Nothing in `src/` hard-codes a scientific hyperparameter.

## Commit

Commit message: `phase-1: repo skeleton, validated config, seeding, run metadata, atomic IO, jsonl, import rules`. Tag `phase-1`. Include `uv.lock`. Do not include `data/raw`, `runs/` or any dataset.

---

**STOP.** I am not starting Phase 2. When Phase 1 is done, reply **"PHASE 1 COMPLETE"** with the `make test` output and the `init-run` line. I will then specify Phase 2 (the environment interface, split machinery, and the first three environments). If Gate 0 changed any notation, default, or the MVP/Stretch table (A0.2), tell me what changed and I will update the later phases to match.

**v2.0 reminder:** the MVP boundary is A0.1 — Curator + GRPO + matched compute + calibration + ROI + the four-method Tier-1 comparison + 3 seeds + Tier-1 LOO + one sealed test = the complete project. Everything past Milestone M-2 is stretch and is cut first on schedule slip.
