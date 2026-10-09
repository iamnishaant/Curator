# CURATOR Roadmap v3.0 — from simulator to paper

Cost-Aware Environment Selection for Agentic RL Training of LLMs

- **Date:** 2026-10-09 (rev. 2: single owner, no time constraint) · **Status:** proposed; solo sign-off, then tag `roadmap-v3`.
- **Relationship to v2.0:** v2.0 ([`CURATOR_Implementation_Roadmap.md`](CURATOR_Implementation_Roadmap.md)) stays the technical reference for Parts B–N (contracts, maths, cost spans, ledger, LOO design). This document **replaces** v2.0's scope table (A0), phase map and calendar (A.3/A.4), experiment matrix (R) and compute plan (O.2) where they conflict. Every change it makes is listed in §11 as a decision to record in `DECISIONS.md`.
- **Working mode (rev. 2):** one person does all the work, and calendar time is **not** the binding constraint. Work is ordered by dependency, not by calendar week. The binding constraints are now (a) GPU-hours and (b) the single-reviewer risk: nobody else checks the work, so the frozen protocol and the gates act as the reviewer. The plan is a **milestone ladder** (§6): each level ends in a complete, publishable state, so ambition can grow without leaving a half-finished project.

---

## 0. Executive summary

**Where we are.** v1 Phases 0–4 are done and committed: config, three environments, splits and the sealed-test guard, the simulator and oracles, and the signal engine with its studies. v1 Phase 5 (discounted UCB plus baselines) is written but uncommitted, and **Gate 2 fails** on the junk-share criterion. Everything that needs a GPU (cost meter, TRL/GRPO, calibration, ROI, LOO) has zero code. We are about 5 weeks ahead of the v2 calendar.

**What the 2025–26 literature changes.** Choosing environments with learnability bandits is now a crowded area: SEC (COLM 2026), DUMP, TAC (ICML 2026) and HarnessBandit. A Sept 2026 evaluation platform, DataFlex-RL, found that **no** selection, reweighting or adaptive mixing policy beat uniform GRPO sampling with a paired 95% interval excluding zero (12 seeds, Qwen2.5-7B). A paper whose headline is "our bandit beats uniform on final score" is therefore **neither novel nor likely to replicate**.

**Where CURATOR's novelty actually is.** None of SEC, DUMP, TAC, HarnessBandit or DataFlex-RL does any of the following, and our searches found no other work that does:
1. Allocate by **measured, attributed GPU-dollar cost**: a single-resource budgeted bandit, which is the knapsack ratio rule.
2. Correct a cheap online proxy against a **held-out calibration benchmark**, and flag environments whose proxy is spurious (S5).
3. Output a **validated per-environment ROI leaderboard**: estimated marginal gain per dollar, checked against leave-one-out.

**The five big changes in v3:**
1. **Reframe the claim:** gain per GPU-dollar at matched budget, plus validated ROI, plus spurious-reward detection. We test it in two pre-registered cost regimes, heterogeneous and homogeneous (§3).
2. **Environment portfolio of 6–7 arms with real cost spread** (MATH, MBPP added; free too-hard arm) instead of three equal-cost arms (§4.1).
3. **Beat the actual SOTA allocators:** an SEC-style and a DUMP-style learnability bandit become **MVP baselines**. Both are computable from rollout scores, with no trainer internals (§4.4).
4. **Size the run horizon and calibration for statistical power** before spending GPU money (§4.2, §4.3).
5. **Measure before modelling, and overlap CPU work with GPU runs.**
   - A small GPU pilot (TRL spike, step time, per-arm pass rates and costs) runs **early**, because its numbers set the portfolio, the round horizon and the simulator re-parameterisation.
   - CPU work (calibration, ROI, LOO on the simulator; paper writing) fills the time while long GPU runs execute (§6).
6. **Climb a three-level ladder** (§6.2), now that time allows:
   - **Level 1:** the complete Tier-1 paper.
   - **Level 2:** a real multi-turn tool-use arm with 5–10× cost spread, plus a Tier-2 model at TAC's scale and a TAC-lite baseline.
   - **Level 3:** novelty extensions, i.e. branch-counterfactual ROI, composition with DAPO, and external validation on DataFlex-RL.

---

## 1. Current status (evidence-based; updated after Phase A, 2026-10-09)

| Area | State | Evidence |
|---|---|---|
| Spec, config, seeding, atomic I/O, CI | ✅ | `docs/SPEC.md`, `tests/unit/test_config.py` |
| GSM8K, Countdown, noisy env; hash splits; sealed guard | ✅ | `data/manifests/MANIFEST.sha256`, contract suite |
| Simulator, DP/static/myopic oracles, S-A…S-H | ✅ | `reports/sim/*.json` (oracle beats Uniform by 11–159%) |
| Signal engine (LP-B chosen, thresholds tuned) | ✅ | D-40, D-42, `reports/analysis/*` |
| D-UCB, mixture map, Static/LP/UCB baselines, sweep | ✅ (Phase A done, uncommitted until you say so) | D-44…D-51, `reports/sweeps/tune_ducb.md` |
| Gate 2 (50 evaluation seeds) | 🟡 **criteria a–e pass**; dev target missed (43% vs 75%), analysed in D-65 | `reports/gate2/gate2.md`, `docs/GATES.md` |
| Phase C: SEC/DUMP-style baselines, fair tuning, S-I/S-J, Gate 2'' | ✅ built; ❌ **Gate 2'' fails (i) on S-I** (passes on S-C); diagnosed, Phase D is the fix | `reports/gate2/gate2b.md`, `reports/analysis/curator_diagnosis.md`, D-68…D-71 |
| Phase D: calibrator, targeted calibration, Calibrated Reward Engine | ✅ built and tested; ❌ **CRE not adopted**: Gate 2'' (i) fails on S-I and CRE ≈ targeted Curator; the reward channel has little leverage on the mixture (D-79) | `reports/gate2/cre_gate.md`, `reports/analysis/cre_diagnosis.md`, D-72…D-79 |
| Test suite / lint | ✅ 248 pass / clean | — |
| Cost meter, trainer, calibration, ROI, LOO, viz | ⬜ empty packages | — |

Mean final scores on 50 evaluation seeds 100–149 (tuned configs of D-68, `status_control: hard`):

| | Uniform | Static | LP | Std UCB | SEC-style | DUMP-style | Curator |
|---|---|---|---|---|---|---|---|
| S-A | 0.579 | 0.608 | 0.610 | 0.624 | 0.578 | 0.572 | **0.640** |
| S-B | 0.405 | 0.418 | 0.417 | 0.421 | 0.413 | 0.413 | 0.416 |
| S-C (cost-heterogeneous) | 0.342 | 0.344 | 0.346 | 0.359 | 0.342 | 0.344 | **0.403** |
| S-I (long horizon, expensive arms are valuable) | 0.356 | 0.352 | 0.369 | **0.372** | 0.365 | 0.364 | 0.356 |
| S-J (S-I with equal costs) | 0.355 | 0.346 | 0.372 | **0.384** | 0.367 | 0.367 | 0.365 |

Reading (updated after Phase C): Curator's advantage is large where cost varies independently of value (S-C, +0.06 over SEC/DUMP-style) and absent in S-I, where the valuable arms (MATH, MBPP) are the expensive ones and the noisy arm soaks up 20–25% of compute before calibration. Standard UCB is the strongest baseline in S-I and S-J. A privileged spurious-reward flag lifts Curator to the SEC/DUMP level in S-I, so calibration (Phase D) is both the project's novelty and the specific fix. **Cost is the lever only when cost and value are unrelated; held-out calibration is what makes it safe otherwise.**

---

## 2. Competitive landscape (checked 2026-10-09)

| Method | Arm signal | Cost-aware | Held-out calibration | Validated attribution | Needs trainer internals |
|---|---|---|---|---|---|
| Graves 2017 / TSCL 2020 | learning progress | ✗ | ✗ | ✗ | ✗ |
| Online Data Mixing (Albalak 2023, pretraining) | loss, EXP3 | ✗ | ✗ | ✗ | ✗ |
| MoDoMoDo 2025 (multimodal RLVR) | offline mixture → outcome regression | ✗ | offline proxy runs | ✗ | ✗ |
| DUMP 2025 | mean \|advantage\|, UCB over distributions | ✗ | ✗ | ✗ | ✗ (from rewards) |
| SEC 2025/26 | mean \|advantage\|, TD(0) bandit | ✗ | ✗ | ✗ | ✗ (from rewards) |
| TAC 2026 | learnability + gradient-alignment transfer | ✗ | ✗ | ✗ | ✓ (projected grads) |
| HarnessBandit 2026 | \|advantage\| + gradient-sketch transfer | ✗ | ✗ | ✗ | ✓ |
| PCL / DAPO dynamic sampling | prompt-level difficulty filtering | indirectly (fewer wasted rollouts) | ✗ | ✗ | ✗ |
| **CURATOR (v3)** | LP + richness, **÷ measured cost**, D-UCB | **✓** | **✓ (C2 regression, S5)** | **✓ (ROI vs LOO)** | ✗ |

**Lessons we adopt:**
- **From DataFlex-RL:** use paired seeds and many of them. Pre-register one benchmark summary, because rankings flipped (ρ = −0.33) between a math-heavy summary and the full suite. Expect parity with uniform when costs are homogeneous.
- **From Spurious Rewards (Shao et al., ICML 2026):** random rewards raise *Qwen* math scores but not Llama/OLMo scores. Our noisy arm is therefore a probe, not a known-useless arm, and the noisy-arm result **must** be replicated on a second model family.
- **From TAC:** transfer matters, and TAC's main baseline is a learnability-only bandit. That is exactly the SEC-style baseline we add. A TAC-lite comparison is in Level 2 (§6.4), because it needs gradients.
- **From TinyZero (to verify in our pilot):** Qwen2.5-0.5B reportedly fails to learn Countdown. Base pass rates per arm must be measured before the portfolio is frozen.

Sources are listed at the end of this document.

---

## 3. Research claims, reframed

### 3.1 What "beating the state of the art" means here

We will **not** beat state-of-the-art *models*. With 0.5–1.5B models and about 100 GPU-hours, nobody can, and saying so protects the paper. We **will** compete with state-of-the-art *allocation methods* under a matched GPU-dollar budget. The claim we aim for:

> Under a fixed compute budget and a cost-heterogeneous environment portfolio, CURATOR reaches a higher held-out score per GPU-dollar than uniform, static, learning-progress and learnability-bandit (SEC/DUMP-style) allocation. It identifies spurious-reward environments online, and its ROI leaderboard agrees with leave-one-out contributions. When costs are homogeneous it matches uniform, consistent with DataFlex-RL.

The second sentence makes the paper robust to a null in the homogeneous regime: we explain **when** adaptive allocation pays off, not merely **that** it does.

### 3.2 Contributions (paper-level)

1. **Budgeted environment allocation for RLVR.** The scheduler maximises the final score at budget B with a gain-per-measured-dollar reward. That reward is the bang-per-buck rule of single-resource bandits-with-knapsacks (Badanidiyuru et al.), which is optimal for the fractional knapsack. It comes with a span-based cost meter and per-environment cost attribution.
2. **Two-timescale calibrated reward.** A cheap proxy (LP, richness) is corrected every K rounds by Bayesian windowed-regression credit (C2/C2d) on a held-out calibration benchmark. Spurious-reward detection (S5) runs online.
3. **Validated ROI leaderboard.** Per-environment estimated marginal gain per dollar comes with bootstrap intervals and is validated against LOO. The C2d transfer matrix comes as a by-product.
4. **A protocol contribution.** A simulator with known optima, pre-registered gates, and a matched-budget ledger that charges calibration and overhead to the method using them.

### 3.3 Pre-registered hypotheses

| ID | Hypothesis | Test (primary) | Where |
|---|---|---|---|
| H1 | Curator > Uniform on score at budget B, **heterogeneous** regime | paired diff, 95% bootstrap CI > 0, 5 seeds | Tier 1 |
| H2 | Curator > best learnability bandit (SEC/DUMP-style) on score at B, heterogeneous regime | paired CI > 0 | Tier 1 |
| H3 | Cost normalisation is what drives H1/H2 | `cost_exponent: 0` ablation loses the gain | Tier 1 + sim |
| H4 | Calibration detects the noisy arm (S5) and cuts its compute share | S5 within the budget in ≥ 2/3 seeds; share drop vs `calib.enabled: false` | Tier 1, 2 model families |
| H5 | ROI ranking agrees with LOO | Kendall τ ≥ 0.4 (N ≥ 5); both junk arms in bottom 2 | Tier 1 |
| H6 | **Homogeneous** regime: Curator ≈ Uniform | equivalence: \|Δ\| < δ_S (pre-registered) | Tier 1 (small) |
| H7 | Discounting helps under drift | sim S-E/S-H only | Tier 0 |

Multiple comparisons: H1 and H2 are co-primary, with a Holm correction. Everything else is secondary.

**Plan for negative results** (written now so it can't be bent later): if H2 fails, the paper becomes "cost-aware calibration and attribution for RLVR environment portfolios". Contributions 2–4 plus H1/H3/H4/H5 still stand. A null in H6 is the *expected* result.

---

## 4. Design changes (v3 deltas)

### 4.1 Environment portfolio v3 (resolves "cost-aware is untestable" and "N = 3 makes LOO meaningless")

| Arm | Domain | Role | Expected cost / prompt | New work | Sealed test |
|---|---|---|---|---|---|
| `gsm8k` | math | learnable, cheap | 1× (~250 tok) | have | GSM8K test |
| `math35` | math | learnable, harder, **long** | ~2–3× (≤1024 tok) | `math-verify` checker **[verify]** | MATH-500 |
| `mbpp` | code | learnable, **expensive** (sandbox) | ~2–4× | sandbox (subprocess, timeout, mem limit) | MBPP test |
| `countdown` | procedural | learnable or hard, model-dependent | ~1–1.5× | have | seed-range test |
| `noisy` | (GSM8K inputs) | spurious-reward probe | 1× | have | — |
| `too_hard` | procedural | zero-signal arm (Countdown 6 numbers, large targets) | ~1.5× | config variant only | — |
| `kk` (first cut) | logic | contamination-free procedural | ~1.5× | reasoning-gym generator **[verify]** | seed-range test |

- **Level 1 core = the first 6 arms.** `kk` joins in Level 2 (§6.4).
- **Homogeneous regime** (for H6) = `gsm8k`, `countdown`, `noisy`, `too_hard`, all with the same token caps.
- **Level 2 adds `toolcall`:** a deterministic multi-turn function-calling environment (calculator, unit conversion, date arithmetic, in-memory lookup API; v2 D.2), so multiple turns and tool executions give a **5–10× cost spread**. That is the regime the title's "agentic" promises, and where the cost lever should be strongest.
- **Calibration slices**, one per domain: math (GSM8K calib + MATH calib), code (MBPP val, pooled because small), countdown, plus logic if `kk` is added.
- **Acceptance rule for the portfolio** (from the pilot, §6 Phase D):
  - ≥ 2 arms with base pass@8 in [0.15, 0.85];
  - `too_hard` with pass@16 ≤ 0.02;
  - measured cost spread (max/min unit cost) ≥ 2×.
- If the rule fails on 0.5B, re-run the pilot on 1.5B (§4.6).

### 4.2 Run horizon and budget design rule

The machinery needs rounds. Warm-up takes 2N rounds, the LP-B window is W = 10, S3/S4 entry needs h = 4, and the α/β fit needs `k_min` = 5 windows. With v2 defaults (R = 5, 1.5 GPU-h) a real run is probably only about 15–35 rounds; the simulator already runs only about 25. **Rule, fixed after the Phase D step-time measurement:**

```
T_rounds = floor(B_train / (R · c_step))  ≥ 60
n_windows = floor(T_rounds / K)           ≥ 12
calib cost = n_windows · c_eval           ≤ 0.10 · B
```

Levers, in order of preference:
- R = 2 steps per round;
- vLLM generation (shorter c_step);
- smaller calibration slices with paired items;
- warm-up = N rounds instead of 2N;
- a larger B.

All simulator scenarios are re-parameterised to the same T_rounds (sim-to-real, Part F.7), and Gate 2 is re-run at that horizon.

### 4.3 Calibration redesign for statistical power

Per-window gains will be small (roughly 0.5–2 points), while an independent 200-item slice has SE ≈ 3.5 points. Calibration therefore only works if:

1. **Items are paired** across windows, so the variance comes only from discordant items: SE(ΔS) ≈ √(d/n), with d = fraction of items that flipped.
2. **Credit pools windows** (C2/C2d regression), never a single-window ΔS.
3. **Generation is fast** (vLLM greedy) so that 12+ windows fit inside 10% of B.

A **Calibration Power Study** runs on the simulator (scenario S-G already sweeps n_b and K) before any GPU calibration run. It fixes K, slice sizes and `k_min` by the criterion "C2 recovers planted m_j with R² ≥ 0.8, and S5 flags noisy within budget in ≥ 90% of seeds". It writes to `reports/analysis/calibration_power.{json,md}`.

### 4.4 Baselines v3 (fair and current)

| # | Baseline | MVP? | Implementation note |
|---|---|---|---|
| 1 | Uniform | ✓ | have |
| 2 | Static, size-proportional (pre-registered) | ✓ | **fix:** the current sim Static equals Uniform (equal π_d), so it tests nothing |
| 3 | LP curriculum (Graves/TSCL, \|LP\| Boltzmann) | ✓ | have |
| 4 | **SEC-style**: TD(0) bandit on mean \|A\| | ✓ **(promoted)** | for binary GRPO rewards with k of G correct, mean \|A\| = 2√(p(1−p)), p = k/G, and 0 when k ∈ {0, G}; needs `sum_abs_adv` in `EnvRoundObs` (schema +1 field) |
| 5 | **DUMP-style**: UCB over mean \|A\| | ✓ **(promoted)** | same signal, UCB rule |
| 6 | Standard UCB on Curator's proxy | ✓ | have |
| 7 | Curator ablations: no-cost, no-calib, no-status | ✓ | config flags |
| 8 | TAC-lite (LoRA gradient-sketch transfer) | Level 2 | trainer internals |
| 9 | Curator + DAPO dynamic sampling (composition) | Level 3 | prompt-level filter inside each arm |

v2 said SEC "needs trainer-internal advantage statistics". That is wrong for GRPO: the advantage is the group-normalised reward, which is computable from rollout scores.

**Fair tuning (Part J.1, currently violated):**
- Only Curator was swept; LP and UCB reuse Curator's τ and ε.
- In v3 every adaptive method gets the same tuning budget: 8 grid points × tuning seeds 0–9, on the same scenarios.

### 4.5 Scheduler fixes (Phase A outcome — supersedes the first draft of this section; Phase C additions in §6.3)

The first draft of this section blamed an "S1 re-entry pump" for the Gate 2 junk-share failure. Tracing and measuring in Phase A showed that was wrong; what actually happened:

1. **The failing Gate 2 report on file was stale.** It was produced before the D-51 fix (established S3/S4 arms no longer fall back to S1). With the current code the criterion passed even before any further change on 20 seeds.
2. **S1 re-entry pump: measured and negligible** (D-66): 0.00% / 0.72% / 0.68% of the budget on S-A / S-B / S-C over 50 tuning seeds. No redesign.
3. **Junk share at 50 seeds** failed under `soft` (38/50 seeds) because a D-UCB bonus surge occasionally put weight on the too-hard arm. `status_control: hard` (S4 capped at the floor, as Roadmap E.5 defines) fixed it on tuning seeds (49/50) and confirmed on evaluation seeds (48/50) (D-64).
4. **New finding, open (D-67): the allocation is close to one-hot.** After warm-up, 97% (S-A), 61% (S-B) and 96% (S-C) of rounds put more than half the mixture on a single arm under `hard` (mean max-weight 0.89 / 0.56 / 0.87). The simulator has no forgetting or between-round interference, so it rewards this; real GRPO could be hurt. Phase C adds a concentration diagnostic and a concentration constraint to the fair-tuning grid.
5. **Status priority** restored to E.5 (S1 over S5 for fresh arms; S3/S4 incumbents exempt from the S1 test; S5 still overrides them), with four regression tests (D-51).
6. **Cost noise in the proxy** (hypothesis for Curator < UCB in S-B, where costs are nearly equal): still open for Phase C — check that `cost_exponent: 0` closes the S-B gap, and consider shrinking c̃ toward 1 when the cross-environment cost CV is below the measurement CV.
7. **Static baseline** fixed (D-63): size-proportional on declared sizes; it was identical to Uniform before.

### 4.6 Model choice

| Role | Model | Why |
|---|---|---|
| Tier-1 main | Qwen2.5-0.5B-Instruct **or** Qwen2.5-1.5B-Instruct (decided by the Phase D pilot) | 0.5B may not learn Countdown or MBPP; 1.5B costs about 3× more per step |
| Noisy-arm replication (H4) | Llama-3.2-1B-Instruct **[verify licence/access]** | spurious-reward gains are Qwen-specific |
| Tier 2 (Level 2) | Qwen2.5-1.5B/3B or Qwen3-1.7B | matches TAC's scale for comparability |

**Pilot rule:** pick the smallest model that satisfies the portfolio acceptance rule (§4.1).

**Decision (Kaggle, rev. 3): Level 1 uses Qwen2.5-0.5B-Instruct.** Reasons:
- Kaggle T4s have 16 GB and no bf16, so 0.5B in fp16 with LoRA is comfortable and 1.5B is tight once rollouts and the reference model are counted.
- Every GPU number triples at 1.5B, and the weekly quota is the binding constraint.
- 0.5B-Instruct should satisfy the §4.1 acceptance rule on GSM8K, MATH (levels 3–5) and MBPP. Countdown is the likely failure (small models often learn it poorly), so it is expected to be the `too_hard`-like or low-signal arm, and the Countdown variants may need easier settings.
- **Escalation rule:** if the Phase B pilot finds fewer than 2 arms with base pass@8 in [0.15, 0.85], switch Level 1 to 1.5B and use the Plan B matrix.
- 1.5B+ belongs to Level 2, and needs extra quota (see §4.9).

### 4.7 Generation backend

Decide in the TRL spike: HF generate vs TRL's vLLM colocate mode. v2 deferred vLLM to after Gate 5 for cost-attribution simplicity. But the horizon rule (§4.2) and calibration cost (§4.3) probably **require** vLLM. Attribution then uses the token-share model (H.4) with the Gate 5 cross-check. Record the choice in `docs/TRL_SPIKE.md`.

### 4.8 Evaluation protocol

- **Primary metric (pre-registered):** macro-average over the domain test sets at budget B. Domains are weighted equally, and `noisy`/`too_hard` have no test set. Secondary metrics: compute-to-target, AUC of score vs GPU-dollar, per-domain scores, and OOD on MATH-500 if `math35` is not trained on.
- **Seeds:** 5 paired seeds for the main heterogeneous matrix and 3 for ablations, the homogeneous regime and LOO.
- **Full-suite reporting:** every run reports every domain. No post-hoc subset summaries (the DataFlex-RL lesson).

### 4.9 Kaggle platform plan (added rev. 3)

Facts to **[verify in your account]**: sessions run up to 12 h; the weekly GPU quota is not officially published (about 30 h is commonly reported, and the exact number is shown in your account settings); the accelerator is **one P100 or two T4s** per session; linking Colab Pro adds 15–30 h/week.

| Constraint | Consequence | Plan |
|---|---|---|
| T4 has **no bf16** | fp16 mixed precision for training and generation | pin fp16 in the trainer config; watch for loss scaling issues in the pilot |
| P100 is not supported by current vLLM (compute capability 6.0) **[verify]** | the P100 is HF-generate only | **choose the 2×T4 option** |
| 2×T4 in one session | two independent GPUs | run **two seeds in parallel, one per GPU** (two single-GPU processes, no multi-GPU training code). This roughly doubles throughput and keeps training single-device |
| Weekly quota (~30 h) | L1 ≈ 125 single-GPU-h ≈ ~63 session-hours with two parallel seeds ≈ 2–3 weeks of quota, if the quota counts session time rather than per-GPU time **[verify]** | schedule matrix blocks per week; finish pilot first to replace all estimates |
| Sessions can end or be interrupted (12 h cap) | runs must resume | checkpoint at every round boundary to `/kaggle/working`; between sessions, persist checkpoints and logs as a Kaggle Dataset version (working dir limit ≈ 20 GB **[verify]**); the kill-and-resume test (Gate 3) is a hard requirement |
| Shared, non-isolated machines | wall-clock noise breaks the cost-CV < 10% gate (H.7) | record GPU type, driver and CPU count per run; run the drift monitor (identical micro-batch every N steps); if CV stays above 10%, fall back to a token-based cost model (attributed tokens × fitted $/token), validated against wall-clock, and say so in `COST_MODEL.md` |
| Hardware differences change costs | costs in GPU-seconds are only comparable on the same GPU type | **all runs in a matrix use T4**; never mix P100/T4 within a comparison |
| Internet may be off by default | model and package downloads fail | enable internet in notebook settings; cache the model and wheels as Kaggle Datasets |
| 16 GB per T4 | vLLM colocate must fit next to training | use `vllm_gpu_memory_utilization` ≈ 0.3, small `max_completion_length`, `num_generations` G = 8 **[verify against pinned TRL]**; fall back to HF generate (slower) if it does not fit |

Development stays on your Windows machine (CPU work); Kaggle runs only the GPU blocks, launched from a notebook that clones the repo at a tagged commit. The repo's CPU-only invariant (v2 O.1) means the same code runs in both places.

---

## 5. Close-out checklist for the current code (Phase 5) — DONE 2026-10-09 (commit/tag pending owner go-ahead)

| # | Item | File(s) |
|---|---|---|
| 1 | Fix 4 failing tests (`gamma` 0.90 asserts; hash test must override to a value ≠ base; S1/S5 priority per decision) | `tests/unit/test_config.py`, `tests/unit/test_status.py` |
| 2 | `ruff check --fix` and the 2 manual fixes (F841 in `gate2_check.py`, E702 in `test_mixture.py`) | — |
| 3 | Write D-44 … D-51 (cited in code, missing in the log) plus the S5>S1 priority and the L1→L1 import-rule relaxation | `docs/DECISIONS.md` |
| 4 | Diagnose and fix the S1 re-entry pump (§4.5) with a regression test | `signals/status.py`, `tests/unit/test_status.py` |
| 5 | Static baseline: size-proportional pre-registered weights | `experiments/run_sim.py` |
| 6 | Re-run Gate 2 on **50** evaluation seeds, oracle-gap target included (done: a–e pass; dev target analysed in D-65) | `experiments/gate2_check.py` |
| 7 | Delete `scratch_debug5/6/7.py` | repo root |
| 8 | `PHASE5_README.md` + `docs/phase_5_explanation.md`, linked from README | docs |
| 9 | Gate 0 leftovers: `GATES.md`, `EXPERIMENT_PROTOCOL.md` (δ margins), `COST_MODEL.md` skeleton written. **Still needs the owner:** one dated entry confirming D-1 … D-15 (then tag `spec-v1`) and confirming the proposed D-52…D-62 | docs, git |

Exit: suite green, ruff clean, Gate 2 report on 50 seeds committed (pass, or fail with a documented analysis), tag `phase-5`.

---

## 6. Phase plan v3 — single owner, dependency-ordered

### 6.1 Working rules for a one-person project

1. **One phase at a time on the critical path. GPU wait-time is CPU time.**
   - While a GPU job runs (pilot, uniform seeds, main matrix), work only on things that cannot change that job's protocol: simulator studies, the ROI/LOO code, the paper text and figures.
   - Never edit trainer code while its runs are in flight.
2. **The protocol is your reviewer.** With no teammate to catch mistakes:
   - every gate's criteria are committed **before** the gate is run;
   - every GPU matrix starts from a hashed, clean-tree freeze (Gate 10 discipline at every level);
   - every number in the paper comes from a script.
   - Run `/code-review` on each phase's diff before tagging it.
3. **Each phase ends green:** tests and ruff pass, a `phase_N` explanation doc exists, a tag is pushed, scratch files are removed (your existing workflow).
4. **Measure before you model.** Real-hardware numbers (step time, per-arm pass rates and unit costs) are collected early and fed back into the simulator. Without them, simulator tuning optimises a world that isn't yours.

### 6.2 Milestone ladder

| Level | What it proves | Ends at | Publishable as |
|---|---|---|---|
| **L1 — Core paper (= the MVP)** | H1–H7 on Tier 1, both cost regimes, SEC/DUMP baselines, calibration, ROI vs LOO, sealed test, noisy probe on two model families | M-2 | strong workshop paper / solid course project |
| **L2 — Agentic + scale** | Same claims with a real multi-turn tool-use arm (5–10× cost spread), Tier 2 at TAC's scale, a TAC-lite baseline, `kk` arm | M-3 | main-track submission |
| **L3 — Novelty extensions** | Branch-counterfactual ROI (cheap local counterfactuals validated against LOO), Curator + DAPO composition, external validity on DataFlex-RL | M-4 | main-track with extra contributions / follow-up paper |

Climb a level only after the previous milestone is tagged. Each level re-freezes its protocol (Gate 10) before its GPU matrix.

### 6.3 Level 1 — ordered phases

Gate criteria not restated here are those of v2 Part Q.

**Phase A — Phase 5 close-out** (CPU). The §5 checklist. **Gate 2'** = the v2 Gate 2 criteria at 50 seeds. Tag `phase-5`.

**Phase B — TRL spike, cost meter, pilot** (first GPU contact; small, about 10 GPU-h)
- **Tasks:**
  - pin TRL/transformers/peft/torch versions;
  - answer the G.2 spike questions and choose HF vs vLLM colocate; write `docs/TRL_SPIKE.md`;
  - build `cost/{meter,ledger,gpu_stats,model}.py` with an injectable clock, plus `scripts/cost_calibration.py`;
  - **pilot:** base pass@8 / pass@16 and unit cost per arm (existing arms now, new arms after Phase E), step time c_step, and calibration-eval time c_eval; on 0.5B and, if needed, 1.5B.
- **Exit:**
  - unit-cost CV < 10%, unattributed time < 3%;
  - model chosen by the §4.1 acceptance rule;
  - horizon parameters (R, K, B) fixed by §4.2;
  - decisions D-56 and D-60 recorded.
- **CPU work while pilot jobs run:** Phase C.

**Phase C — SOTA baselines, fair tuning, realistic simulator** (CPU) — **built 2026-10-09; Gate 2'' failed on S-I, see below**
- **Tasks:**
  - add `sum_abs_adv` to `EnvRoundObs` and emit it from `world.py`;
  - implement `SECStyleBandit` and `DUMPStyleUCB`;
  - fair tuning for all adaptive methods (`experiments/sweeps/tune_all.py`);
  - re-parameterise every scenario to the measured horizon;
  - add **S-I**, a sim analogue of the v3 portfolio using pilot costs and pass rates, and **S-J**, its homogeneous twin.
- **Gate 2''** (50 evaluation seeds, paired):
  - (i) Curator > SEC-style and DUMP-style in S-C and S-I, CI > 0;
  - (ii) Curator ≥ Uniform − δ in S-J;
  - (iii) the original Gate 2 criteria still hold;
  - (iv) junk share passes (or a documented analysis in D-18 style).
- **Stop rule:** if (i) fails, the cost lever is not working. Diagnose here, where it is cheapest; do not build GPU integration on top of it.
- **Outcome (D-68 … D-71):** all tasks done except "re-parameterise every scenario to the measured horizon" (needs Phase B's step time; S-I/S-J use placeholders at about 70 rounds). Gate 2'': (i) passes in S-C and fails in S-I (−0.008 vs SEC and DUMP), (ii)–(iv) pass. The stop rule fired and the cause was diagnosed (`reports/analysis/curator_diagnosis.md`): the noisy arm takes 20–25% of compute pre-calibration; cost normalisation hurts when valuable arms are expensive; the best static mixture depends on GSM8K↔MATH transfer. **Disposition:** (i) on S-I is not relaxed. It becomes the acceptance test for Phase D, which must pass it with the calibrated Curator and no privileged information. A privileged S5 flag (upper bound only) brings Curator to the SEC/DUMP level in S-I, not to Standard UCB.
- **Implication for the paper:** the expected headline is not "Curator v0 beats everything". It is a two-part claim: cost-aware allocation wins where cost varies independently of value, and calibration is what keeps it safe where cost and value are correlated. Standard UCB is the baseline to beat on S-I/S-J-like portfolios.

**Phase D — Calibration, ROI, LOO on the simulator** (CPU)
- **Tasks:**
  - `calibration/{regression,credit}.py`: C1, C2, C2d, the α/β fitter (ridge-to-prior, non-negative, trust region, leave-one-window-out CV, null-model fallback), and the S5 detector wired to `SignalEngine.set_calibration_mismatch`;
  - `roi/{engine,bootstrap,leaderboard}.py`;
  - `evaluation/loo.py`;
  - the Calibration Power Study (§4.3) and `ab_estimator_study.py`.
- **Gates (simulator parts):**
  - **6-sim:** R² ≥ 0.8 on planted α/β, ≥ 20% better than naive, S5 on the noisy arm within budget in ≥ 90% of seeds;
  - **7-sim:** Spearman vs true marginal ROI ≥ 0.8, junk last in ≥ 90%, interval coverage ≥ 90%;
  - **9-sim:** LOO within ±10% of true contribution.
- **New acceptance test from Phase C (D-69):** re-run `experiments/gate2b_check.py` with the calibrated Curator; criterion (i) on S-I must pass with no privileged information. The upper bound to aim at (not a target) is the privileged-S5 row of `curator_diagnosis.md`.
- **Output:** K, slice sizes and `k_min` frozen.

**Phase E — Environment expansion** (CPU, then a short GPU pilot)
- **Tasks:** `envs/math35.py` (math-verify), `envs/mbpp.py` + `envs/sandbox.py`, the `too_hard` Countdown variant, manifests and the contract suite. Then re-run the Phase B pilot on the new arms.
- **Gate 1B':**
  - gold answers pass ≥ 99%, garbage ≤ 1%;
  - the sandbox kills an infinite loop;
  - manifests reproduce exactly;
  - MinHash near-duplicate check passes;
  - `too_hard` pass@16 ≤ 0.02;
  - measured cost spread ≥ 2×.

**Phase F — GRPO baseline and Curator integration** (GPU, about 15 GPU-h)
- **Tasks:** `trainer/{trl_adapter,reward_wrapper,prompt_stream,callback,round_controller,checkpoint}.py`; uniform × 3 seeds; passthrough equivalence; realised-share test; kill-and-resume test; attribution cross-check.
- **Gates:** **3, 4, 5** (v2 criteria; δ margins are already in `EXPERIMENT_PROTOCOL.md`).
- **CPU work while seeds run:** `viz/` figure functions on fixture logs; paper §2–§4 drafts.

**Phase G — Real calibration, sim-to-real refit, freeze** (GPU, about 10 GPU-h)
- **Tasks:**
  - Curator runs with calibration on; check Gate 6 real;
  - refit the simulator from pilot and uniform logs (v2 F.7) and re-confirm Gate 2'' on the refit;
  - write `docs/ANALYSIS_PLAN.md`.
- **Milestones:** **M-1**, then **Gate 10 (L1 freeze).**

**Phase H — L1 main matrix** (GPU, §7)
- Run exactly as frozen. **Gate 8.**
- **CPU work meanwhile:** the analysis scripts and `make report` on partial logs. Never peek at the sealed test.

**Phase I — LOO, sealed test, paper** (GPU about 15 GPU-h, then CPU)
- LOO (L-C variant) on 4 arms including both junk arms; one sealed-test access per final checkpoint; all Part S figures; the paper; the Part U checklist.
- **Gate 9 → M-2. Tag `L1`.** The project is complete at this point; everything after is additive.

### 6.4 Level 2 — agentic arm, scale, TAC-lite

**Phase J — `toolcall` environment.** A deterministic multi-turn function-calling generator. Each episode runs several turns, and the tool executions are charged to the `SCORE` span.
- The trainer needs a multi-turn rollout path: TRL `rollout_func` or an environment-loop wrapper **[verify against the pinned TRL]**.
- Contract suite plus the Gate 1B' criteria; target cost spread ≥ 5× vs GSM8K.
- Add `kk` (reasoning-gym) here as well.

**Phase K — TAC-lite baseline.**
- Learnability (mean |A|) plus a transfer term from cosine similarity of per-arm LoRA gradient sketches (random projection, as in TAC/HarnessBandit).
- It lives only in `trainer/` (it needs gradients) and is exposed to the L1 scheduler through an extra optional `EnvRoundObs` field. The import rules stay intact.
- Validate on a simulator analogue first, using a planted transfer matrix as truth.

**Phase L — Tier 2 matrix** (Qwen2.5-1.5B/3B or Qwen3-1.7B).
- Pilot, then Gate 10 (L2 freeze).
- Matrix: Uniform, SEC-style, TAC-lite, Curator, Curator-no-cost; 3 seeds; heterogeneous portfolio including `toolcall`.
- LOO on 3 arms × 2 seeds.
- **M-3. Tag `L2`.**

### 6.5 Level 3 — novelty extensions

**Phase M — Branch-counterfactual ROI** (the strongest extra contribution)
- At calibration window k, fork the checkpoint. Train n short steps on the current mixture without arm i, and n steps with it, then compare calibration-slice gains.
- This gives a *local, interventional* marginal-value estimate at a fraction of full LOO cost.
- **Validate:**
  - first in the simulator, where exact counterfactuals exist;
  - then against the L1 LOO results.
- **Use:** as an unbiased correction term for C2 credit, and as a second validation of the leaderboard.

**Phase N — Composition with DAPO dynamic sampling.**
- Prompt-level filtering of all-0 and all-1 groups runs inside each arm, with Curator choosing between arms.
- Test whether the gains add up, with cost charged for the oversampled rollouts.
- This answers the obvious reviewer question: "does it still help on a modern GRPO recipe?"

**Phase O — External validity on DataFlex-RL** **[verify code release and license]**
- Implement Curator as a DataFlex-RL data policy and run it under their protocol at the largest scale you can afford.
- Even a partial replication against their uniform baseline is strong evidence, because their paper is the field's sceptical benchmark.

**Phase P — verl adapter** (optional). Implement the same `TrainerAdapter` protocol (v2 G.6).

---

## 7. Experiment matrix v3 and compute plan

Planning assumption: one Tier-1 run = B ≈ 1.0 GPU-h on an L4/A10-class GPU with vLLM. **Replace it with the Phase B measurement.**

### Level 1

| Block | Methods | Seeds | Runs | GPU-h |
|---|---|---|---|---|
| Pilots, cost calibration, model choice | — | — | ~10 | 10 |
| Uniform baseline + integration (Gates 3–5) | — | 3 | ~6 | 15 |
| Real calibration (Gate 6) | Curator | 3 | 3 | 5 |
| Main, heterogeneous regime | Uniform, Static, LP, SEC-style, DUMP-style, Std-UCB, Curator | 5 | 35 | 35 |
| Ablations | Curator no-cost / no-calib / no-status | 3 | 9 | 9 |
| Homogeneous regime (H6) | Uniform, SEC-style, Curator | 3 | 9 | 9 |
| Second family, noisy probe (H4) | Uniform, Curator on Llama-3.2-1B | 2 | 4 | 4 |
| LOO (L-C, 4 arms) | Curator minus arm i | 3 | 12 | 12 |
| Sealed test + reporting evals | — | — | — | ~4 |
| **L1 total** | | | | **≈ 105 (+20% ≈ 125)** |

### Level 2 (Tier 2 ≈ 3–4 GPU-h per run at 1.5B; ×2 at 3B)

| Block | Runs | GPU-h |
|---|---|---|
| Tier-2 pilot + toolcall pilot | ~6 | 15 |
| TAC-lite validation on Tier 1 | 3 | 3 |
| Tier-2 main (5 methods × 3 seeds) | 15 | 50–60 |
| Tier-2 LOO (3 arms × 2 seeds) | 6 | 20–25 |
| **L2 total** | | **≈ 90–100 (+20% ≈ 115)** |

### Level 3

About 20–40 GPU-h for branch counterfactuals (short forks) and DAPO composition. DataFlex-RL scale depends on their setup.

**Grand total ≈ 260–280 GPU-h for all three levels.** If GPU access is limited:
- finish L1 first; it needs about 125 GPU-h;
- the **Plan B** for L1 (≈ 60 GPU-h) is B = 0.6 GPU-h per run, 5 main methods × 4 seeds, ablations no-cost and no-calib only, homogeneous regime with 2 seeds, LOO on 3 arms.

---

## 8. Analysis plan essentials (to be frozen at Gate 10)

- Paired differences per seed; 10,000-draw percentile bootstrap; Holm correction across the co-primary H1/H2.
- Equivalence (H6, Gate 4): \|Δ\| < δ, with δ_S = 1.0 benchmark point registered in `EXPERIMENT_PROTOCOL.md`. Non-significance is never treated as equivalence.
- No claim when intervals overlap. Report steps, rollouts and tokens per method as covariates.
- ROI wording: "estimated marginal contribution under the mixtures actually run", never causal.
- Every number in the paper comes from one script (`make report`).

---

## 9. Paper plan

- **Working title:** *Paying for What Teaches: Cost-Aware, Calibrated Environment Allocation for RLVR.*
- **Structure:**
  1. Introduction: the cost-heterogeneity problem and the DataFlex-RL null.
  2. Related work: §2 table.
  3. Method: budgeted D-UCB, two-timescale calibrated reward, ROI.
  4. Simulator study with known optima.
  5. Tier-1 results in both cost regimes.
  6. ROI vs LOO.
  7. Spurious-reward detection on two model families.
  8. Limitations: K.3, sim-to-real, small models, 2–4× cost spread for single-turn arms.
- **Headline figures:**
  - score vs GPU-dollar, both regimes;
  - paired differences vs each baseline;
  - weights over rounds with status timeline;
  - ROI vs LOO scatter;
  - S5 detection timeline on Qwen vs Llama.
- **Venue by level:** L1 → a NeurIPS/ICLR workshop on efficient or RL post-training. L2 (agentic arm + Tier 2 + TAC-lite) → a main-track submission (ICLR/ICML/NeurIPS or COLM, where SEC appeared). L3 strengthens the main-track version or becomes a follow-up.

---

## 10. Top risks

| Risk | Trigger | Fallback |
|---|---|---|
| Cost lever too weak in real arms | Pilot spread < 2× | Long-completion MATH arm; pull the L2 `toolcall` arm forward into L1 |
| 0.5B cannot learn the new arms | Pilot: < 2 arms in [0.15, 0.85] | Switch Tier 1 to 1.5B (about 3× compute, so Plan B matrix) |
| Calibration too noisy | Power study fails at the feasible K | Pool slices, larger K with fewer windows, C1 fallback; H4 tested on sim + Tier 1 descriptively |
| Curator ≤ learnability bandit (H2 null) | Gate 2'' (i) fails | Diagnose in sim first; the paper pivots to contributions 2–4 (§3.3) |
| TRL hook drift | Spike finds brittle hooks | Pin the version; in-house GRPO loop (v2 G.1 option D) |
| GPU access insufficient | < 125 GPU-h available for L1 | L1 Plan B (≈ 60 GPU-h); L2/L3 only with more compute |
| Spurious-reward confound on Qwen | Noisy arm raises the score | That is a finding: report it with the Llama replication |
| Single-owner blind spots (no second reviewer) | Any gate run without pre-committed criteria; any figure without a script | §6.1 rules: criteria committed before runs, `/code-review` per phase, freeze before every GPU matrix |
| Scope creep across levels | Starting L2 work before `L1` is tagged | Ladder rule (§6.2): climb only after the previous milestone is tagged |

---

## 11. Decisions to record (proposed IDs, after sign-off)

- **D-52** Claim reframed to gain per GPU-dollar plus calibration plus validated ROI; two cost regimes (§3).
- **D-53** Portfolio v3: 6 Level-1 arms, `kk` and `toolcall` in Level 2; MATH and MBPP promoted from stretch (§4.1).
- **D-54** SEC-style and DUMP-style baselines promoted to MVP; the v2 "needs trainer internals" rationale is withdrawn (§4.4).
- **D-55** Fair tuning budget for all adaptive methods (§4.4).
- **D-56** Horizon rule T ≥ 60 rounds, ≥ 12 windows, calibration ≤ 10% (§4.2).
- **D-57** Calibration uses paired items and is sized by the power study (§4.3).
- **D-58** 5 seeds for the main matrix; Holm on H1/H2 (§4.8, §8).
- **D-59** Second model family for the noisy probe is in the MVP (§4.6; was a Part U checklist item without a plan).
- **D-60** Generation backend chosen in the spike; vLLM allowed before Gate 5 (§4.7).
- **D-61** Size-proportional static mixture pre-registered (§4.4).
- **D-62** Single owner; dependency-ordered phases replace the calendar; milestone ladder L1/L2/L3, with v2's stretch items promoted into L2/L3 (§6).

---

## 12. Open questions (answer before Phase B)

Answered in rev. 3:
- **Hardware:** Kaggle, 2×T4 sessions (§4.9).
- **Model:** Qwen2.5-0.5B-Instruct for Level 1, with the escalation rule in §4.6.

Still open (check in the Kaggle account, 5 minutes):
1. What is the **actual weekly GPU quota** shown in Settings, and does the Colab Pro link apply to you?
2. Does a 2×T4 session's quota count session time (good) or per-GPU time?

---

## Sources (literature check, 2026-10-09)

- [DataFlex-RL: An Evaluation Platform for RLVR Data Policies (arXiv 2609.06107)](https://arxiv.org/pdf/2609.06107)
- [TAC: Transferability for General Reasoning — An Automated Curriculum for Multi-Domain RLVR (arXiv 2606.25178)](https://arxiv.org/pdf/2606.25178) · [ICML 2026 listing](https://icml.cc/virtual/2026/78178)
- [HarnessBandit (arXiv 2609.13739)](https://arxiv.org/pdf/2609.13739)
- [RISED: multi-environment selection (arXiv 2610.00979)](https://arxiv.org/html/2610.00979v1)
- [SEC: Self-Evolving Curriculum for LLM Reasoning (arXiv 2505.14970)](https://arxiv.org/abs/2505.14970v4)
- [DUMP: Distribution-Level Curriculum Learning (arXiv 2504.09710)](https://arxiv.org/pdf/2504.09710)
- [MoDoMoDo: Multi-Domain Data Mixtures for Multimodal RLVR (arXiv 2505.24871)](https://arxiv.org/pdf/2505.24871)
- [Spurious Rewards: Rethinking Training Signals in RLVR (ICML 2026)](https://icml.cc/virtual/2026/poster/61082)
- [Bandits with Knapsacks (Badanidiyuru et al.)](https://arxiv.org/pdf/1305.2545)
- Prompt Curriculum Learning (PCL): [review](https://liner.com/review/prompt-curriculum-learning-for-efficient-llm-posttraining)
