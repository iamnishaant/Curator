# DECISIONS LOG

One entry per decision. Teammates append; entries are never deleted.

Format:

```
## D-<number> — <short title>
- Date:
- Owner:
- Decision:
- Reason:
- Affects:
```

---

## D-1 … D-15 — Specification defects S-1 … S-15 (Gate 0)

Status at repo creation: **resolved by the Roadmap v2.0 spec-review table (top of the roadmap document)**. At Gate 0 each teammate confirms their read-through of the affected section; confirmations are recorded here as entries D-1 … D-15 with date + initials. Any *change* to a resolution gets a new entry.

## D-16 — Dashboard wording aligned with v2.0 scope

- Date: (Gate 0)
- Owner: team
- Decision: proposal says "offline experimental dashboard" (figures + HTML leaderboard); live weight/cost monitoring moves to stretch.
- Reason: MVP scope freeze (Roadmap v2.0 A0.5); avoids a proposal/implementation mismatch during evaluation.

## D-17 — verl moved from promised plug-in to planned extension

- Date: (Gate 0)
- Owner: team
- Decision: proposal says "initial implementation through TRL, with verl support planned as an extension".
- Reason: TRL API is still moving and integrations dominate risk budget; verl adapter is Roadmap stretch S-15.

## D-18 — Gate 2 oracle-gap closure is a development target, not a pass threshold

- Date: (Gate 0)
- Owner: team
- Decision: Primary gate criteria are paired improvement over Uniform (S-A), no underperformance (S-B), junk share, status metrics; ≥75% oracle-gap closure is a target. Results in 60–75% with tight intervals and seed-consistent behaviour pass with a documented analysis here.
- Reason: a hard threshold on a noisy stochastic simulator would fail good implementations on noise.

## D-19 — Gate 4 equivalence via pre-registered margins δ, not KS p-values

- Date: (Gate 0)
- Owner: team
- Decision: register equivalence margins (final score |Δ| < δ_S, per-step reward |Δ| < δ_R) before the passthrough runs; KS checks are diagnostics only.
- Reason: "p > 0.05" is absence of evidence, not evidence of equivalence.

## D-20 — Split salt frozen

- Date: 2026-10-02 (Phase 2)
- Owner: P2/P3
- Decision: `data.split_salt = curator-gate0-2026-10-d20` in `configs/base.yaml`. Hash formula exactly as SPEC s3: `sha256(salt + dataset + problem_id)`.
- Reason: the salt must be frozen before any split is built; manifests are committed anyway, but freezing the salt at Gate 0 makes pre-Gate-0 reruns reproducible.

## D-21 — Environment parameters centralized in `configs/base.yaml` (data.envs.*)

- Date: 2026-10-02 (Phase 2)
- Owner: P2
- Decision: per-environment parameters (token limits, verifier timeout, Countdown generator ranges, noisy mode/q/flip_p, cost priors) live in `configs/base.yaml` under `data.envs.<env>` instead of the Roadmap's `configs/env/*.yaml` files. No `configs/env/` files are created.
- Reason: `data.envs.*` is part of the pydantic-validated `RootConfig`, so every environment parameter is inside the config hash and unknown-key/mis-typed validation applies. Separate per-env YAML files would duplicate keys and risk drift between the two sources.
- Affects: Roadmap Phase 2 "Files to create" list (configs/env/*) — deviation is intentional and documented here per the SHOULD-override rule.

## D-22 — Countdown split seed ranges

- Date: 2026-10-02 (Phase 2)
- Owner: P3
- Decision: procedural Countdown instance seeds are drawn from disjoint ranges train [0, 1e7), calib [1e7, 1.5e7), dev [1.5e7, 2e7), test [2e7, 3e7); constants in `envs/countdown.py::SPLIT_SEED_RANGES`, disjointness checked by leakage tests.
- Reason: Roadmap D.2 fixes the boundaries at 1e7/2e7 but leaves the calib/dev boundary open; 1.5e7 keeps ranges equal-sized.

## D-23 — DP oracle (O1) implemented as a vectorised skill-grid DP with conservative cost rounding

- Date: 2026-10-03 (Phase 3)
- Owner: P1
- Decision: `simulator/oracle.py::DPOracle` is exact backward induction over a per-env skill grid (default 80 levels, sized from the maximum reachable skill) × budget units (default 100), numpy-vectorised. Restrictions enforced at construction: N ≤ 3, no drift, no late-start envs, `cost_skill_slope == 0`. Costs round UP to budget units. Per-round skill gains quantise to whole grid levels (numpy `rint`), which is pessimistic for sub-level gains. `select_mixture` replans from the live world state each round (receding horizon, shared value table).
- Reason: a first recursive float-key DP memoised on (skills, units) exploded combinatorially (> 5 min/episode); a coarse level grid *stalls* (sub-quantum per-round gains never accumulate because states are re-decoded each round). The grid DP is feasible and, with up-rounded costs, provably dominates every fixed mixture under the same discretised model (unit-tested via `fixed_policy_value`). Roadmap A0.2 marks the DP oracle optional; the restrictions keep its state space Markov.
- Affects: Phase 3 item 10 unit test semantics (dominance is stated within the discretised model, not against continuous dynamics).

## D-24 — Static oracle (O2) uses random search + hill-climb for N > 4

- Date: 2026-10-03 (Phase 3)
- Owner: P1
- Decision: N ≤ 4 exhausts the simplex grid (step 0.1); N > 4 draws 1500 seeded Dirichlet candidates and hill-climbs 300 pairwise perturbations.
- Reason: the full grid at N = 8 has C(17,7) = 19448 points per scenario-seed; random search + refinement reaches comparable values for a fraction of the work. CMA-ES was rejected as an extra dependency.

## D-25 — Myopic oracle (O3) is a first-order equimarginal rule

- Date: 2026-10-03 (Phase 3)
- Owner: P1
- Decision: `MyopicOracle` sets w_i proportional to max(0, dS/ds_i · eta_i_eff · g_i / c_i); zero for noisy and not-yet-available envs.
- Reason: it implements the equimarginal rule to first order; second-order and cross-env transfer effects are ignored. Validated as a strong reference (42–74% gap closure across S-A..S-H), not an exact optimum.

## D-26 — `MixtureDecision` deferred to Phase 5; Phase 3 schedulers return plain weight dicts

- Date: 2026-10-03 (Phase 3)
- Owner: P1
- Decision: `BaseScheduler.select_mixture` returns `dict[str, float]` with a `validate_weights` helper (sum = 1, non-negative, keys = env ids). The `MixtureDecision` type (ucb_scores, intents, statuses…) is added in v1 Phase 5 when D-UCB produces those fields.
- Reason: nothing in Phase 3 consumes the richer type; adding it now would be an unused abstraction.

## D-27 — Random baseline module named `random_baseline.py`, not `random.py`

- Date: 2026-10-03 (Phase 3)
- Owner: P1
- Decision: `scheduler/baselines/random_baseline.py`.
- Reason: avoids any ambiguity with the stdlib `random` module in tooling that resolves bare module names.

## D-28 — Scenario configuration is a standalone strict schema, not part of `RootConfig`

- Date: 2026-10-03 (Phase 3)
- Owner: P1
- Decision: `configs/sim/*.yaml` files are validated by pydantic models in `simulator/scenarios.py` (`ScenarioCfg`, extra="forbid") and are NOT merged into `configs/base.yaml` / `RootConfig`. Simulator-only knobs (budget, R/P/G, benchmark items, rollout concentration, oracle resolution) live there.
- Reason: scenario worlds are experiment fixtures, not scientific hyperparameters of the real system; mixing them into `RootConfig` would bloat the frozen config hash of every run and create unused-key noise for GPU runs. Same unknown-key-fails guarantee as the main config.
- Affects: Roadmap Phase 3 "Files to create" list (configs/sim/*.yaml kept, but schema location deviates deliberately).

## D-29 — Import-rule test fixed to recognise `curator_rl.<pkg>` submodule paths

- Date: 2026-10-03 (Phase 3)
- Owner: P1
- Decision: `tests/architecture/test_import_rules.py` now records the first TWO components for `curator_rl.*` imports; L1 modules may import `curator_rl.core` and their own package `curator_rl.<self>`.
- Reason: the Phase 1 parser kept only the top-level component, so `from curator_rl.core.types import X` looked like a bare `curator_rl` import and the L1-only-core exception could never fire. Latent until Phase 3 added the first L1 code (scheduler).
