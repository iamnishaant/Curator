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

## D-30 — `EnvStatus` + `SignalVector` added in Phase 4; `MixtureDecision` still deferred

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: `core/types.py` gains `EnvStatus` (StrEnum: S1_unexplored … S5_unreliable) and the frozen `SignalVector` dataclass (posterior pass rate + CI, lp/lp_se/lp_z, lp_raw_fast/slow, richness, mean score, unit cost / cost-norm, proxy_raw/proxy_unit/proxy_reward, status + S2 sub-label note). `MixtureDecision` stays deferred to v1 Phase 5 per D-26.
- Reason: these are the first types with a Phase 4 producer and consumer; adding MixtureDecision now would still be an unused abstraction.

## D-31 — Richness smoothing: per-round fraction EW with `signals.lambda`

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: `RichnessEstimator` smooths the per-round informative fraction with the signal lambda (stored value unchanged on zero-prompt rounds); it does not pool counts Beta-style.
- Reason: round fractions carry the heterogeneity directly; counts would need a separate Beta prior parameterisation that is not in the roadmap spec for SR.

## D-32 — Band/variance richness fidelity under the round aggregate

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: the observation aggregate carries only `n_groups_mixed` (0<k<G). Band membership is estimated from the posterior [p_lo, p_hi] overlap with the band; the variance mode decides on the discounted per-round group-rate std and emits 0/1. `RichnessEstimator.update_groups` provides the exact per-group path (used by unit tests and wherever raw groups exist later).
- Reason: Roadmap S-8 collapses the 0.1–0.9 band to `mixed` for G = 8, so the aggregate path is exact for the MVP settings; the per-group entry point keeps fidelity possible without schema churn.

## D-33 — Interim unit-cost EMA inside the Signal Engine

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: `c_i` for the proxy is an EMA of `EnvRoundObs.cost_usd / n_prompts` computed by the engine (updated only on prompt-bearing rounds; unknown unit cost => c_norm = 0 => proxy reward 0). The real Cost Meter (Phase 6) replaces the source behind the same interface.
- Reason: cost-normalised rewards are needed from Phase 5 on; the interim source keeps the proxy testable off-GPU. Never `estimate_cost()` priors.

## D-34 — S5 mismatch input API; full wiring deferred to Phase 9

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: `StatusClassifier.set_mismatch(flag)` + `SignalEngine.set_calibration_mismatch(env_id, flag)` carry the streak counters and S5 enter/leave (q, q' from `calib.mismatch_windows` / `mismatch_clear_windows`). Phase 4 never feeds real flags — the α/β fitter that determines mismatch does not exist yet.
- Reason: the state machine is built and unit-tested now; the informational dependency is genuinely Phase 9 calibration.

## D-35 — Proxy discount reuses `signals.lambda`

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: σ_LP and the r̃ reservoir are discounted with `signals.lambda`; no separate `proxy.lam` field.
- Reason: one less unconfigurable scale; the roadmap leaves the proxy discount unspecified and a distinct value would need its own tuning grid.

## D-36 — Dwell semantics: gates S3/S4 exits; S2 exits ungated

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: leaving S3/S4 requires the hysteresis leave-rule AND `dwell_min` completed rounds; leaving S2 is ungated (entries into S3/S4 already require h consecutive rounds); leaving S1 is exempt and entering S5 is exempt (Roadmap E.5); leaving S5 uses q' clean windows.
- Reason: reading "D_min before leaving ANY status" literally would make S2's exit the slowest path in the machine with no benefit — the residual class needs no protection. The S3-exit classification falls through to _from_s3_exit (S2 by default, S4 only if its entry rule is already mid-run) so a forgetting spike cannot re-enter S3.

## D-37 — LP-B uses two-stage WLS with residual-variance inflation; LP-C runs on raw per-round rates

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: LP-B's SE divides the weighted residual spread by (k − 2) and inflates each round's variance by that spread (stage-2 weights = 1/(se_i² + resid_var)); LP-C's window means use the RAW per-round rate (k/n of that round alone), not the discounted pooled posterior.
- Reason: the first LP-B SE divided by Σw·(k−2), shrinking the SE by the prompt count and giving a measured FPR of 1.0 (flat sequences flagged significant 100% of the time). LP-C on pooled windows overlaps mechanically (both windows share EW mass) and measured FPR 0.40; on raw rates ≈ 0.06.

## D-38 — Studies collect episodes in-process via a recording scheduler

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: `experiments/analysis/common.py` wraps Uniform/Random in a `RecordingScheduler`, runs `run_episode` directly and keeps the full `RoundObservation` stream plus the harness log's `true_skills`/`true_pass_rates` in memory. No change to the harness, world or `run_sim.py`.
- Reason: `run_sim.py` does not persist round logs; rebuilding worlds/episodes deterministically keeps the studies bit-reproducible without new plumbing.

## D-39 — Truth status labels mirror the prediction-side S1 evidence band

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: ground-truth S1 = the same discounted group-evidence state machine the predictor uses (n_groups_eff below the entry floor / above n_min), computed from allocations only (bookkeeping, not a hidden signal). S2/S3/S4 truth uses the true pass rate and true skill velocity; the S3 truth velocity anchor is 1e-2 skill/round; SR truth = informative-band heuristic.
- Reason: an undiscounted `rounds_seen`-only truth mismatched the prediction semantics for starved arms (1031 bogus errors in validation) and would have made every F1 number meaningless.

## D-40 — LP estimator study result: LP-B chosen; rule applied as pre-registered

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: LP-B (weighted slope, W=10) is adopted (`signals.lp_method: lp_b`). Study numbers (S-A/S-B/S-E × {uniform, random}; tuning seeds 0–19, evaluation 100–129): LP-B rho 0.19–0.20 vs true skill velocity, FPR 0.000, lag 3, sign accuracy 0.70–0.72. LP-A had better rho (0.23–0.27) but FPR 0.103–0.104 > the 5% constraint; LP-C rho 0.13–0.16 and sign accuracy 0.31. Both phases picked LP-B.
- Reason: the roadmap's pre-registered selection rule (best rho subject to FPR ≤ 5% and lag ≤ 5; fallback LP-A) selects LP-B, not the expected fallback. LP-A remains implemented (config), and the FPR miss is documented rather than silently allowed.

## D-41 — Study flip-rate estimator: pooled post-first-exit rate

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: `flip_rate_pooled` = total transitions after each env's first S1 exit × 100 / total post-exit intervals, pooled across envs and captures. Used instead of a mean of per-env rates (which a late-S1-exit arm with a 2-interval tail can blow up to 100).
- Reason: the roadmap's "transitions per environment per 100 rounds" is a rate whose per-env mean is not a stable estimator on short tails; the pooled version measures exactly the flapping quantity with full precision.

## D-42 — Tuned status thresholds replace the roadmap starting values

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: `configs/base.yaml` status block becomes p_sat 0.7, p_hard 0.05, z_up 1.5, dwell_min 3, consecutive_rounds 4 (h); sr_hard 0.10, z_neg 2.0, hysteresis 0.05 unchanged. NEW config key `signals.status.s1_entry_ratio: 0.5` — S1 is re-entered only below half of n_min evidence (exit still needs n_min), a roadmap-native anti-flap constant (the S1 exploration quota already references n_min/2).
- Reason: tuning grid (243 points over p_sat/p_hard/z_up/dwell/h × S-A/S-B/S-E/S-G) maximised truth-anchored macro-F1 subject to pooled flip ≤ 5: winner F1 0.9705/flip 0.37 on tuning seeds 0–9; validated on evaluation seeds 100–119: F1 0.9672/flip 0.38 (eval's own argmax differs only in h=2 vs 4 with near-identical F1; h=4 kept as the tuning-phase pick). Exit criteria met: F1 ≥ 0.85, flip ≤ 5/100. The pre-flip numbers (per-env-rate flip 6–7.5, bound-driven S1 re-entries) motivated D-41's estimator and the S1-entry hysteresis.

## D-43 — Truth anchor frozen as constants inside the tuning study

- Date: 2026-10-05 (Phase 4)
- Owner: P1
- Decision: `experiments/analysis/tune_status_thresholds.py` labels ground truth with hardcoded TRUTH_ANCHOR constants (the pre-tuning roadmap defaults), NOT the current base.yaml values; the grid varies prediction thresholds only.
- Reason: anchoring truth to mutable base.yaml values would make any post-hoc re-run circular (truth moving with the tuned prediction).
