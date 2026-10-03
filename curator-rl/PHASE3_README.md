# PHASE 3 README — The Curator Simulator: World, Oracles, Scenarios, Harness

This document describes **everything Phase 3 of the CURATOR project built,
how it works, and why each piece is the way it is**. Phase 2 gave us the
data layer (environments, splits, sealed guard). Phase 3 builds the
**algorithm testbed**: a synthetic world whose true optimum is known, the
oracles that compute that optimum, the scenario suite the scheduler will be
validated on, and the episode harness that will later be re-used almost
unchanged for tuning and ablations.

> The formal contract lives in `docs/SPEC.md`, the full plan in the Roadmap
> v2.0 (Part F, Part P Phase 3). Deviations from the Roadmap are logged in
> `docs/DECISIONS.md` (D-23 … D-29). A shorter companion write-up is
> `docs/phase_3_explanation.md`.

---

## 1. Why Phase 3 exists at all

CURATOR's core claim is that *adaptive, cost-aware allocation beats fixed
allocation under a matched compute budget*. Before any GPU money is spent,
three things must be established, and only the simulator can establish them:

1. **It is the only place the correct answer is known** (Roadmap Rule 3 /
   Decision 2). Real runs cannot tell us the optimal mixture — the simulator
   can, because we wrote the dynamics. Every scheduler decision path (learn,
   saturate, drop junk, revive a hard arm) is exercised against ground truth
   before Phase 7 touches a GPU. This is why Phase 3 is a *buffer phase* on
   the critical path: it gates all scheduler work.
2. **The scheduler must be developed against the exact interface it will
   use in production.** The simulator emits the same `RoundObservation` /
   `CalibrationObservation` types (Roadmap B.3) the real trainer will emit in
   Phase 8. Scheduler code validated off-GPU therefore transfers unchanged;
   a GPU dependency anywhere in L1 or the simulator even fails the
   import-rule test (O.1 invariant, CPU-only development).
3. **Fair tuning needs a free playground.** γ, κ, τ, ε and the status
   thresholds are tuned on simulator *tuning seeds* (E.7), and the Tier 0
   sweeps run 50+ seeds per scenario because they cost CPU-minutes, not
   GPU-hours.

Everything below serves one of those three goals.

---

## 2. What we built, file by file

```
src/curator_rl/core/types.py          + EnvRoundObs, RoundObservation, CalibrationObservation
src/curator_rl/scheduler/base.py      BaseScheduler interface + weight validation
src/curator_rl/scheduler/baselines/   Uniform and Random stubs (harness validation)
src/curator_rl/simulator/world.py     SimWorld + SimEnvParams (the world model)
src/curator_rl/simulator/quotas.py    weights -> integer prompt quotas
src/curator_rl/simulator/scenarios.py Scenario schema (strict pydantic) + builders
src/curator_rl/simulator/oracle.py    DPOracle (O1), StaticOracle (O2), MyopicOracle (O3)
src/curator_rl/simulator/harness.py   run_episode, budget stop rule, gap-closure metric
src/curator_rl/simulator/replay.py    Trace-replay skeleton (Tier 1R, stretch)
configs/sim/scenario_s?.yaml          The eight scenarios S-A..S-H
experiments/run_sim.py                CLI: methods x seeds, JSON reports
tests/unit/test_sim_world.py          World-model unit tests (Phase 3 item 10)
tests/unit/test_oracle.py             Oracle unit tests
tests/unit/test_quotas.py             Quota invariants
tests/unit/test_scheduler_baselines.py Uniform/Random validity
tests/integration/test_sim_scenarios.py  End-to-end scenario runs (item 11)
reports/sim/S-*.json                  Generated validation evidence
docs/phase_3_explanation.md           Companion write-up
```

### 2.1 `core/types.py` — the observation contracts (Roadmap B.3)

| Type | What it carries | Why it exists |
|---|---|---|
| `EnvRoundObs` | per-env round statistics: prompts, rollouts, successes, `n_groups_mixed` (groups with 0 < k < G), score sums, tokens, verifier/GPU seconds, cost | The unit the Signal Engine (Phase 4) consumes. Group-level richness counting and score moments are included now so Phase 4 needs no schema churn |
| `RoundObservation` | `round`, `steps`, `per_env`, `weights_used`, `round_cost_usd`, `overhead_usd`, `budget_remaining_usd` | Everything the scheduler may see. **Nothing else**: true skills and true pass rates live only on `SimWorld` and never enter the observation (F.1 — the scheduler must not read hidden variables) |
| `CalibrationObservation` | window index, benchmark score by domain with SEs, charged eval cost, `exposure_by_env` (compute mass per env since the last calibration) | Produced by the harness now so the Phase 9 calibration consumer plugs into a working schema; `exposure_by_env` is the W_{j,k} that the C2 credit regression (Part I.2) will fit against |

`MixtureDecision` and `RoiRecord` still wait: defining a type before its
consumer exists invites churn (D-26 records the Phase 3 decision to return
plain weight dicts from `BaseScheduler`).

### 2.2 `scheduler/base.py` — the pure scheduler interface (Roadmap A.2, Decision 1)

`BaseScheduler` has exactly one required method:

- `select_mixture(observation | None) -> dict[str, float]` — the mixture for
  the next round; `None` before the first round.

Everything else is an optional hook: `update_observation`,
`update_calibration`, `get_state`/`save_checkpoint`/`load_checkpoint` (the
checkpoint pair arrives with real state in Phase 5, Part M item 13).

**Why so small:** the scheduler is a *pure function of observations*. It
never imports torch, TRL, datasets, or the simulator; the import-rule test
enforces that L1 (`scheduler`) may import only `core`. This is what makes
the scheduler testable, replayable on traces (F.6), and portable to TRL or
verl. `validate_weights` enforces the mixture invariant (keys = env ids,
non-negative, sum 1) so every method and test shares one definition of a
legal mixture.

The Uniform and Random baselines in `scheduler/baselines/` exist now not as
competition but as **harness validation**: if `w_i = 1/N` cannot run an
episode end-to-end, the harness is broken; if Random (Dirichlet(1) per
round, seeded via a named `SeedManager` stream) behaves like Uniform on
average, the seeding is sound.

### 2.3 `simulator/world.py` — the world model (Roadmap F.2)

Each environment has a **latent skill** `s_i`; everything observable is
derived:

```
p_i      = sigmoid(a_i * (s_i - d_i))          true pass probability
g_i      = 4 * p_i * (1 - p_i)                 learnability (peaks at p = 0.5)
s_j(t+1) = s_j(t) + R * eta_j * sum_i T_ji * w_i * g_i(t) + xi
cost/round = M * sum_i w_i * c_i(t)
```

**Why these formulas:** `g_i` is the GRPO signal analogue — a prompt group
whose pass rate is near 0 or near 1 carries (almost) no advantage signal,
exactly like 4p(1−p). This makes "saturated" and "too hard" arms *emerge*
from the dynamics instead of being hand-labelled, which is precisely what
the scheduler must learn to detect from noisy observations.

`SimEnvParams` (one dataclass, all environment *types* are parameter
settings, not code — F.2):

| Parameter | Meaning / env type it creates |
|---|---|
| `skill0, slope a, difficulty d` | starting point and shape of the pass-probability curve. Low `d` → easy/saturating; high `d` → too-hard (`p ≈ 0`, `g ≈ 0`) |
| `eta` | per-round learning rate (scaled by R steps/round) |
| `cost_usd_per_prompt` | mean unit cost; `cost_lognormal_sigma` adds per-prompt noise |
| `transfer_out: {target: T}` | positive = transfer of learning, negative = interference (S-D). Self-transfer `T_ii = 1` is implicit; transfers *into* noisy envs are rejected — a noisy arm must stay useless (F.2) |
| `noisy_q` | noisy type: reported successes ~ Binomial(G, q) **independent of skill**; `eta` forced to 0; skill frozen |
| `gate: {prereq, theta, slope}` | delayed type: `eta` gated by `sigmoid(slope * (s_prereq − theta))` — zero learning until the prerequisite's skill crosses the wall |
| `drift: {round, shift}` | abrupt regime change: difficulty shifts once after the given round (S-E) |
| `late_start_round` | env is unavailable before this round; the harness zeroes its weight and renormalises (S-H cold start) |
| `cost_skill_slope` | cost-drifting type: unit cost rises with skill |
| `bench_slope, bench_difficulty, bench_weight` | the env's own benchmark slice: `σ(a_b(s − d_b))`, weight π_d (all None → uniform) |

`SimWorld.step_round(weights)`:

1. converts weights to integer prompt quotas (2.4),
2. computes `g_i` from the *current* skills, applies the skill update with
   Gaussian noise `xi` (the world is stochastic),
3. applies one-shot drift at round boundaries,
4. draws rollouts per prompt: `q ~ Beta(κ_b·p, κ_b·(1−p))` — per-prompt
   heterogeneity, so groups are *mixed* the way real GRPO groups are (a bare
   Binomial count would make the richness signal degenerate) — then
   `k ~ Binomial(G, q)`; noisy envs draw `k ~ Binomial(G, noisy_q)` directly,
5. draws costs with log-normal noise and assembles the `RoundObservation`.

**Noise is bounded by construction:** `p` is clamped away from 0/1 before
the Beta draw; `xi` size is a scenario knob (`skill_noise_std`).

`evaluate_benchmark(observed=True)` draws `Binomial(n_b, p_d)` per domain
slice and pools by π_d with propagated SEs — benchmark measurement is
*noisy and chargeable*, which is what makes the choice of `n_b` and `K`
(Kalibration interval) a real design question (scenario S-G).
`observed=False` returns the noiseless expectation — **oracle-only**.

### 2.4 `simulator/quotas.py` — weights to prompts (Roadmap D.4)

`largest_remainder_quotas(weights, M)` converts a mixture into integer
prompt counts: floor, then distribute the remainder by largest fractional
part with a **seeded jitter tie-break**, so `sum(m_i) == M` exactly and the
mapping is deterministic. This is the same rule the Phase 8 prompt stream
will use inside the real trainer — building it here, tested here, means the
trainer reuses code that already has provenance.

### 2.5 `simulator/scenarios.py` — the scenario suite (Roadmap F.3)

A scenario is a *parameter setting*, not code. `ScenarioCfg` is a strict
pydantic model (extra="forbid", so a typo in a YAML fails loudly) and
validates cross-references: duplicate env ids, gate prerequisites that don't
exist, transfer targets that don't exist. `build_world(cfg, seed)` draws the
concrete world through `SeedManager(seed).rng("sim_world")` — the same
(scenario, seed) pair always yields the same world, and the world's
randomness is independent of the scheduler's stream.

| ID | N | Contents | The question it answers |
|---|---|---|---|
| S-A | 3 | easy/saturating, valuable-slow, noisy | Does an allocator drop the saturated and the noisy arm? |
| S-B | 8 | full portfolio analogue incl. noisy, too-hard, gated, interfering | Realistic overall comparison (Gate 2 criterion b) |
| S-C | 4 | identical dynamics, costs 1×..8× | Value of cost normalisation (hypothesis H2) |
| S-D | 4 | helper→core positive transfer, interferer→core negative transfer | Credit rules C1 vs C2, interference handling (H5) |
| S-E | 3 | one env's difficulty shifts abruptly at round 12 | Value of discounting/adaptation (H4) |
| S-F | 3 | too-hard arm unlockable via transfer from the normal arm | Starvation vs revival (exploration floor) |
| S-G | 8 | S-B world with noisy benchmark slices (n_b = 50) + calib enabled | Choice of calibration set size and K |
| S-H | 5 | valuable env arrives at round 8; weak early portfolio | Cold start |

**Why exactly these eight:** they cover every scheduler decision path the
MVP needs — learn, saturate, junk, cost heterogeneity, transfer,
interference, drift, revival, cold start (A0.4: "three environments … and
the simulator scenarios cover the harder dynamics"). Parameters were chosen
so each scenario *discriminates*: the oracle beats Uniform by > 10% in all
of them (§4) — a scenario where uniform ≈ oracle cannot rank methods and
would be redesigned (Phase 3 failure condition 13).

### 2.6 `simulator/oracle.py` — the known optima (Roadmap F.4)

All oracles share the same mean-dynamics evaluator,
`deterministic_final_score`: a noiseless episode (no rollout noise, no cost
noise, no `xi`) under a mixture policy, returning the expected benchmark.
It works on **copies** of the env parameters — oracles read the live world
by design, but must never mutate it (unit-tested).

**O2 `StaticOracle` — best fixed mixture.**
- N ≤ 4: exhaustive simplex grid, step 0.1 (66 points at N = 3, 286 at
  N = 4).
- N > 4: 1500 seeded Dirichlet draws + 300 pairwise hill-climb steps
  (D-24). The full grid at N = 8 has C(17,7) = 19,448 points per seed;
  random search + refinement reaches comparable values for a fraction of
  the work. CMA-ES was rejected as an extra dependency for marginal gain.

**O3 `MyopicOracle` — first-order equimarginal rule (D-25).** Each round,

```
r_i = dS/ds_i * eta_i_eff * g_i / c_i        w_i proportional to max(r_i, 0)
```

with `dS/ds_i = pi_i * a_b * p_b * (1 - p_b)`. This is "allocate by true
marginal benchmark gain per dollar" to first order; second-order and
cross-env transfer effects are ignored. It is a *strong, cheap reference*
(42–74% gap closure in validation), not an exact optimum — the exactness
claim belongs to O1.

**O1 `DPOracle` — exact DP (D-23, the most design-heavy piece).**
Dynamics are deterministic, so value iteration runs backward over budget
units on a per-env skill grid:

```
V(s, u) = max( bench(s), max_w V(step(s, w), u - cost_units(w)) ),  V(s, 0) = bench(s)
```

The whole table is computed with numpy (backward sweep over units; per
action a precomputed gain table in grid levels, gathered by fancy indexing).

Two earlier designs failed and are recorded here because the reasoning is
the point:

- *Float-key memoised recursion* (states keyed on skills rounded to 1e-3,
  budget units): exploded combinatorially — > 5 min per episode, timeout.
- *Coarse level grid*: **stalls**. A per-round skill gain smaller than one
  level quantum never accumulates, because the transition re-decodes the
  skill to the level value each round — progress above the decoded point is
  discarded every round, so mixed allocations with sub-quantum gains look
  permanently worthless. The DP then underperformed Uniform (−26% gap).

The final design sizes the grid from the **maximum reachable skill**
(span capped at 4.0, default 80 levels) and costs are rounded **UP** to
budget units. Consequences, all deliberate:

- cost rounding up ⇒ the DP never over-credits affordability ⇒ its value is
  *conservative*: it provably dominates **every fixed mixture evaluated
  under the same discretised model** (`fixed_policy_value`, unit-tested);
- gain quantisation (numpy `rint`) is pessimistic for sub-level gains, which
  only under-values nearly-worthless allocations;
- with 80 levels / 100 units the DP's *realised* gap closure in S-A is
  ~79% and its value matches the static oracle's — the earlier 40-unit
  version under-planned (it believed rounds cost more than they do) and
  hit only 28%.

Restrictions are enforced at construction and are not negotiable: N ≤ 3,
no drift, no late-start envs, `cost_skill_slope == 0` — anything else makes
the dynamics time- or cost-state-dependent, which the unit-indexed backward
induction cannot represent. That leaves S-A and S-F as the DP's scenarios;
elsewhere the static oracle is the reference (A0.2 marks O1 as optional
precisely for this reason).

`select_mixture` is a *receding-horizon* policy: each round it replays the
optimal first action from the live world state (the value table is solved
once per episode and then queried). Oracles are references, not schedulers
— they are allowed to read hidden state; the scheduler is not.

### 2.7 `simulator/harness.py` — episodes under the ledger rule (Roadmap F.5, H.6)

`run_episode(scheduler, scenario, seed)`:

1. builds a fresh world for (scenario, seed) — no state leaks between
   episodes;
2. loops: scheduler → weights; `apply_scenario_constraints` zeroes weights
   of not-yet-available (late-start) envs and renormalises; the world steps;
   the crossing round **is charged** and the episode stops right after it —
   the round-granularity version of the H.6/S-13 stop rule (overshoot ≤ one
   round, asserted by `within_budget_tolerance`);
3. every `K` rounds (if `calib.enabled`), produces a charged
   `CalibrationObservation` with the window's `exposure_by_env` and window
   cost — the same accounting the Phase 9 calibration will use;
4. final reporting evaluation is **uncharged** (S-13: reporting evals are
   identical and free for every method);
5. logs every round with weights, costs, per-env outcomes — plus true
   skills and true pass rates as *diagnostics in the log, not in the
   observation*. This is deliberate: we can plot scheduler decisions against
   ground truth without the scheduler ever seeing it.

`oracle_gap_closure` implements the headline metric
`(S_method − S_uniform) / (S_oracle − S_uniform)` at equal budget.

### 2.8 `simulator/replay.py` — the trace-replay skeleton (Roadmap F.6, stretch)

`TraceReplayer` loads per-round per-env outcome chunks from a JSONL trace
and replays a scheduler against them under the **arm-clock assumption** (an
env's outcome depends only on its own number of pulls). When the scheduler
pulls more than the trace did, the last chunk repeats. The docstring carries
the validity warning verbatim: replay is for scheduler logic and
hyperparameters only — **never** for transfer or leaderboard claims. Built
now so Tier-1 logs have a fixed format to land in.

### 2.9 `experiments/run_sim.py` — the experiment entry point

```
python experiments/run_sim.py --scenario configs/sim/scenario_sa.yaml \
    --methods uniform,random,static_oracle,myopic_oracle,dp_oracle --seeds 20
```

Runs any method set × seeds, prints a summary table (mean ± std, rounds,
cost, gap closure) and writes a JSON report to `reports/sim/`. Gap
denominators use the best available oracle (max of DP/static). `--dp-levels`
controls the DP grid (3 learnable dims are O(levels³) — the documented
knob when a 3-env DP is slow).

---

## 3. How it was verified (Phase 3 evidence)

Full suite: **119 tests pass, ruff clean, CPU-only** (O.1 invariant).

### 3.1 World-model unit tests (`test_sim_world.py`, Phase 3 item 10)

- `p ∈ (0, 1)` for skills far off the difficulty; `sigmoid` matches numpy.
- learnability `g` peaks at p = 0.5, is 0 at p ∈ {0, 1}.
- a zero-transfer env never moves other envs' skills, even at w = 1.
- the noisy env's skill is *frozen* under allocation, its reward rate hits
  the design value within 0.02 (Roadmap D.5 #5).
- round cost identity: with cost noise off,
  `round_cost == M · Σ w_i · c_i` exactly; quotas sum to M with
  largest-remainder rounding.
- drift fires once, at the right boundary; gate keeps a gated env at zero
  learning until the prerequisite's skill crosses θ.
- determinism: two worlds from the same seed produce identical rounds.

### 3.2 Oracle unit tests (`test_oracle.py`)

- DP value ≥ every fixed grid mixture under the same discretised model
  (the Phase 3 item 10 dominance property), and ≫ uniform.
- DP ≈ continuous mean dynamics for a no-transfer world (|Δ| < 0.02) —
  quantisation does not distort the answer, only limits resolution.
- static oracle beats uniform and **does not mutate the live world**.
- myopic weights are a valid mixture, give nothing to the noisy arm, and
  prefer the cheaper of two identical arms.
- DP rejects N = 4, drift, and cost-skill coupling with clear errors.

### 3.3 Integration tests (`test_sim_scenarios.py`, item 11)

- all eight scenarios run end-to-end with Uniform and Random under shrunk
  budgets: budget respected, logs complete, weights valid;
- same seed ⇒ bit-identical episode;
- S-G produces and charges calibration observations;
- **exit criterion (item 12)**: in S-A the oracle beats Uniform by > 10
  benchmark points over 5 seeds.

### 3.4 Scenario validation via `run_sim.py` (10 seeds, full budgets)

Oracle-gap closure, `(S_method − S_uniform) / (S_oracle − S_uniform)`:

| Scenario | Uniform | Static oracle | Myopic | Uniform→oracle improvement |
|---|---|---|---|---|
| S-A | 0.574 | 0.722 (gap 100%) | 0.653 (53%) | **+25.7%** |
| S-B | 0.410 | 0.460 (100%) | 0.440 (59%) | **+12.2%** |
| S-C | 0.346 | 0.493 (100%) | 0.412 (45%) | **+42.4%** |
| S-D | 0.411 | 0.481 (100%) | 0.443 (45%) | **+17.1%** |
| S-E | 0.324 | 0.393 (100%) | 0.353 (42%) | **+21.3%** |
| S-F | 0.211 | 0.547 (100%) | 0.458 (74%) | **+159%** |
| S-G | 0.407 | 0.452 (100%) | 0.437 (66%) | **+11.2%** |
| S-H | 0.476 | 0.592 (100%) | 0.535 (51%) | **+24.4%** |

Every scenario discriminates (> 10%); the DP closes ~79% of the S-A gap as
an adaptive policy. Reports are committed as evidence artifacts under
`reports/sim/`.

---

## 4. Documented deviations from the Roadmap

| ID | Deviation | Why |
|---|---|---|
| D-23 | DP oracle = vectorised skill-grid DP with up-rounded costs; restricted to N ≤ 3, no drift/late-start/cost-coupling | Float-key DP exploded; coarse grids stall (sub-quantum gains never accumulate); the restrictions keep the DP state Markov. A0.2 itself marks O1 optional |
| D-24 | Static oracle uses seeded Dirichlet random search + hill-climb for N > 4 | 19,448-point grid per seed is not worth it; refinement reaches comparable values; CMA-ES = extra dependency |
| D-25 | Myopic oracle is a first-order equimarginal rule | Exact marginal-ROI allocation is not closed-form; first order is a strong, cheap reference, not an exactness claim |
| D-26 | `MixtureDecision` deferred to Phase 5; schedulers return plain weight dicts | Nothing in Phase 3 consumes the richer type |
| D-27 | Random baseline module named `random_baseline.py` | Avoids ambiguity with the stdlib `random` module |
| D-28 | Scenario configs are a standalone strict schema (`simulator/scenarios.py`), not part of `RootConfig` | Scenario worlds are fixtures, not scientific hyperparameters; mixing them into the hashed RootConfig would bloat every run's config hash. Same unknown-key-fails guarantee |
| D-29 | Import-rule test fixed to recognise `curator_rl.<pkg>` submodule paths | Phase 1 parser kept only the top-level component, so the L1-may-only-import-core exception could never fire; latent until Phase 3 added the first L1 code |

---

## 5. How to run everything

```bash
pip install -e ".[dev]"

# verification
python -m pytest                        # 119 tests: unit + contract + architecture + integration
python -m ruff check src tests scripts

# simulator experiments
python experiments/run_sim.py --scenario configs/sim/scenario_sa.yaml \
    --methods uniform,random,static_oracle,myopic_oracle,dp_oracle --seeds 20
python experiments/run_sim.py --scenario configs/sim/scenario_sb.yaml \
    --methods uniform,random,static_oracle,myopic_oracle --seeds 20   # N=8: no DP
```

Reproducibility notes:
- every episode is a pure function of (scenario, seed): worlds, scheduler
  streams and oracle streams are independent named `SeedManager` streams;
- the committed `reports/sim/*.json` are regenerated exactly by re-running
  the commands above (the S-* reports in the table used `--seeds 10`).

---

## 6. What Phase 4 is, and why it comes next

Next is the **Signal Engine and status classifier** (v1 Phase 4 / v2 Phase
5-signals): pass rate with discounted pooled counts, the LP estimator study
(A/B/C), signal richness, the proxy reward, and the S1–S5 status logic with
thresholds tuned *on this simulator* (macro-F1 ≥ 0.85 against true status,
flip rate ≤ 5/100 rounds). Phase 3 is what makes that verifiable: the world
knows the true skill and true status, so estimator quality is measurable
instead of assumed — and because the simulator emits the exact
`RoundObservation` stream the Signal Engine will consume in production, the
Phase 4 code plugs into both worlds unchanged.
