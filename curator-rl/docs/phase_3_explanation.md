# Phase 3 — Curator Simulator: what was built and why

**Roadmap reference:** v1 Phase 3 ("Curator simulator"), Part F; v2.0 Phase 3
(MVP, buffer phase — it gates all scheduler work). **Status: complete.**

Phase 3 delivers the only place in the project where the true optimum is
known (Rule 3): a synthetic world with known dynamics, oracles, a scenario
suite, and an episode harness. Phases 4–6 develop and tune the scheduler
against this world before any GPU money is spent (Gate 2).

## What was built, file by file

| File | What it does | Why it exists (Roadmap section) |
|---|---|---|
| `src/curator_rl/core/types.py` (modified) | Adds `EnvRoundObs`, `RoundObservation`, `CalibrationObservation` | B.3 data contracts: the simulator, trace replayer and real trainer must produce the *same* observation schema, so scheduler code validated off-GPU transfers unchanged |
| `src/curator_rl/scheduler/base.py` | `BaseScheduler` (pure observation → mixture) + `validate_weights` | A.2 decision 1: the scheduler is a pure function of observations; it never reads hidden state, torch, or datasets |
| `src/curator_rl/scheduler/baselines/uniform.py` | Uniform baseline (w_i = 1/N) | J.1 #1; also validates the harness and later the static GRPO pipeline (Gate 3) |
| `src/curator_rl/scheduler/baselines/random_baseline.py` | Random mixture (Dirichlet(1) per round) | J.1 #3 sanity floor |
| `src/curator_rl/simulator/world.py` | `SimWorld` + `SimEnvParams`: latent skills, pass probability `p = σ(a(s−d))`, learnability `g = 4p(1−p)`, transfer matrix `T_ji`, per-prompt Beta rollouts (`κ_b`), log-normal costs, benchmark slices with Binomial noise, drift/gated/late-start/noisy/too-hard env types | F.2 world model — every environment *type* is a parameter setting, not separate code |
| `src/curator_rl/simulator/quotas.py` | Largest-remainder rounding: weights → integer prompt quotas summing exactly to M | D.4 batching rule; the real trainer will reuse it in Phase 8 |
| `src/curator_rl/simulator/scenarios.py` | Strict pydantic scenario schema + `load_scenario`/`build_world` (seeded via `SeedManager`) | F.3: scenarios are parameter settings; parameters drawn per seed so nothing is tuned to one hand-built world |
| `src/curator_rl/simulator/oracle.py` | `DPOracle` (O1), `StaticOracle` (O2), `MyopicOracle` (O3), shared mean-dynamics episode simulator | F.4 known optima — the yardsticks for oracle-gap closure |
| `src/curator_rl/simulator/harness.py` | `run_episode` under the budget-ledger stop rule (round granularity), optional charged calibration evaluations, uncharged final reporting eval, per-round logs with hidden diagnostics kept out of observations | F.5 harness; H.6/S-13 stop rule; F.1 isolation (scheduler sees only `RoundObservation`) |
| `src/curator_rl/simulator/replay.py` | Trace-replay skeleton (Tier 1R) with the arm-clock assumption and its caveats in the docstring | F.6 — stretch, but the format is fixed now so Tier-1 logs can be replayed later |
| `configs/sim/scenario_sa..sh.yaml` | The eight scenarios S-A..S-H | F.3 scenario suite, one question each |
| `experiments/run_sim.py` | CLI: methods × seeds per scenario; prints and writes a JSON report with oracle-gap closure | F.5 outputs; the Phase 3 exit criterion is checked with it |
| `tests/unit/test_sim_world.py` | p ∈ (0,1); learnability peaks at 0.5; zero-transfer env never moves other skills; noisy env never moves skill and hits its design rate; round cost = M·Σw_i·c_i; quota rounding; drift/gate semantics; determinism by seed | Phase 3 item 10 (Part M items) |
| `tests/unit/test_oracle.py` | DP dominates every grid mixture; DP ≫ uniform; DP ≈ continuous mean dynamics; static beats uniform and never mutates the world; myopic validity; DP restriction guards | Phase 3 item 10 |
| `tests/unit/test_quotas.py`, `tests/unit/test_scheduler_baselines.py` | Quota invariants; Uniform/Random validity and determinism | Phase 3 item 10 |
| `tests/integration/test_sim_scenarios.py` | All eight scenarios run end-to-end with Uniform and Random; budget respected; determinism; calibration produced and charged; S-A oracle improvement > 10 points | Phase 3 item 11–12 |

## Key design decisions (see `DECISIONS.md` D-23..D-28)

- **The world hides nothing from itself but hides everything from the
  scheduler.** True skills and true pass rates exist only on `SimWorld`;
  observations carry measured quantities. Oracles deliberately break this
  wall (they are references, not schedulers).
- **DP oracle (O1)** is exact backward induction over a per-env skill grid ×
  budget units, vectorised with numpy. A coarse quantisation *stalls* (a
  per-round skill gain smaller than one grid level never accumulates because
  the state is re-decoded each round), so the grid is sized from the maximum
  reachable skill (default 80 levels) and costs are rounded **up** to budget
  units (100), which makes the DP value conservative: it provably dominates
  every fixed mixture evaluated under the same discretised model (unit
  tested). The DP is restricted to N ≤ 3, no drift, no late-start, no
  cost-skill coupling — the cases where the state (skills, budget) is
  Markov. S-A and S-F are its scenarios; everywhere else the static oracle is
  the reference.
- **Static oracle (O2)** exhaustively searches the simplex grid (step 0.1)
  for N ≤ 4; for N = 8 it uses seeded Dirichlet random search + local
  hill-climbing (19k grid points would be the alternative; refinement gets
  within noise of it for less work).
- **Myopic oracle (O3)** is the first-order equimarginal rule: weight ∝
  marginal benchmark gain per dollar. It is a strong, cheap reference
  (42–74% gap closure in validation), not an exact optimum.
- **Observations are schema-identical to the future real system**, including
  token/cost fields (synthetic in the simulator, measured from Phase 7 on).

## How it was verified

1. `pytest` — 119 tests green, including the new unit and integration suites
   (CPU only, per the O.1 invariant).
2. `ruff check` clean.
3. Scenario validation via `experiments/run_sim.py` (10 seeds, full budgets):
   every scenario beats Uniform by **more than 10% relative benchmark score**
   with the static oracle — the Phase 3 exit criterion (item 12):

   | Scenario | Uniform | Static oracle | Myopic | Improvement |
   |---|---|---|---|---|
   | S-A | 0.574 | 0.722 | 0.653 | +25.7% |
   | S-B | 0.410 | 0.460 | 0.440 | +12.2% |
   | S-C | 0.346 | 0.493 | 0.412 | +42.4% |
   | S-D | 0.411 | 0.481 | 0.443 | +17.1% |
   | S-E | 0.324 | 0.393 | 0.353 | +21.3% |
   | S-F | 0.211 | 0.547 | 0.458 | +159% |
   | S-G | 0.407 | 0.452 | 0.437 | +11.2% |
   | S-H | 0.476 | 0.592 | 0.535 | +24.4% |

   Reports: `reports/sim/S-*.json`. The DP oracle additionally closes ~79% of
   the S-A gap as an adaptive policy and exceeds every fixed mixture inside
   its own model.
4. None of the scenarios can be run into a state where uniform ≈ oracle
   (failure condition item 13) — the smallest separation is S-G at +11.2%.

## What Phase 3 deliberately does *not* do

- No Signal Engine, no D-UCB, no status logic (Phases 4–5).
- No MixtureDecision type yet (Phase 5; schedulers currently return plain
  weight dicts).
- Calibration evaluation machinery exists in the harness (charged, producing
  `CalibrationObservation`) but nothing consumes it yet (Phase 9).
- Trace replay is a documented skeleton (F.6 is Tier 1R / stretch).
