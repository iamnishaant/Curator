# PHASE 2 README — Environments, Splits, and the Sealed-Test Guard

This document describes **everything Phase 2 of the CURATOR project built,
and why**. Phase 1 gave us a tested skeleton (config, seeding, run metadata,
atomic I/O). Phase 2 turns that skeleton into the **data layer**: a frozen
environment interface, the first three training environments, deterministic
split machinery, and the guard that protects the sealed test set.

> The formal contract lives in `docs/SPEC.md` (sections 3, 5, 8), the full
> plan in the Roadmap v2.0 (Part D, Part P Phase 2). Decisions that deviate
> from the Roadmap are logged in `docs/DECISIONS.md` (D-20, D-21, D-22).
> A shorter companion write-up is `docs/phase_2_explanation.md`.

---

## 1. Why Phase 2 exists at all

CURATOR's whole thesis is *cost-aware allocation across environments*. Before
we can schedule anything, three things must be true and are impossible to fix
later if wrong:

1. **Every environment reports through one interface.** The scheduler must
   treat GSM8K, Countdown and the noisy arm identically, and later the
   simulator must feed the scheduler exactly the same record types as the
   real trainer (Roadmap Decision 1). So the interface is frozen *first*.
2. **Data splits can never leak.** If calibration or test data reaches
   training, every result the project produces is invalid (Roadmap Part T
   lists leakage as risk #1). Split assignment, manifests and the sealed
   guard therefore exist *before* any training code does.
3. **The noisy environment is a hypothesis test, not decoration.** The
   calibration mechanism (the project's main novelty) exists to catch a
   proxy that rewards noise. Without a noisy arm whose "true" uselessness we
   know, hypothesis H3 has nothing to test.

Everything below serves one of those three goals.

---

## 2. What we built, file by file

```
src/curator_rl/core/types.py         The frozen data contracts
src/curator_rl/envs/base.py          The frozen Environment interface
src/curator_rl/envs/splits.py        Hash-based split assignment + manifests
src/curator_rl/envs/registry.py      Name -> environment factories (config-driven)
src/curator_rl/envs/gsm8k.py         GSM8K environment
src/curator_rl/envs/countdown.py     Countdown environment + safe AST evaluator
src/curator_rl/envs/noisy.py         Noisy-reward environment
src/curator_rl/evaluation/guard.py   SealedTest guard
scripts/download_data.py             GSM8K download with provenance
scripts/build_splits.py              Processed splits + manifests (idempotent)
scripts/env_profile.py               Per-environment throughput/verifier profiles
tests/unit/test_splits.py            Split-assignment + manifest tests
tests/unit/test_guard.py             All sealed-guard refusal paths
tests/contract/test_env_contract.py  The Gate 1 contract suite
configs/base.yaml                    Extended with data.envs.* + frozen salt
reports/env_profiles/phase2_profile.json   Generated profiles (Gate 1 evidence)
```

### 2.1 `core/types.py` — the data contracts (Roadmap B.3)

| Type | What it carries | Why it exists |
|---|---|---|
| `Prompt` | `prompt_id`, `env_id`, `split`, `messages`, `reference` (verifier-only), `meta` | One item flowing from any environment to any consumer. `reference` must never reach the model, so it lives outside `messages` |
| `Verdict` | `success`, `score` ∈ [0,1], `parse_ok`, `verifier_seconds`, `info` | Verification separated from reward shaping. `parse_ok` distinguishes "unparseable answer" from "wrong answer" — a parse failure is a different failure mode for the scheduler's signals |
| `RolloutGroup` | one prompt + its G rollouts (scores, token counts, verifier time) | The unit the Signal Engine consumes later; rollouts within a group are correlated, and LP statistics depend on the group being the unit (Roadmap E.2) |
| `CostPrior` | cold-start cost estimate | Exists explicitly so it can be *forbidden* as a measured cost (Roadmap H: cost is measured, never estimated) |

The observation types (`RoundObservation`, `CalibrationObservation`,
`MixtureDecision`, `RoiRecord`) join this file in the phases that first use
them — defining them now would invite churn before their consumers exist.

### 2.2 `envs/base.py` — the frozen interface (Roadmap D.1, SPEC §8)

Seven methods, each with a contract the tests check:

- `generate_prompt(split, rng)` — deterministic given (split, rng state);
  never serves another split's items.
- `generate_batch(split, n, rng, exclude)` — without replacement within a
  batch; fresh instances for procedural environments.
- `evaluate_response(prompt, response) -> Verdict` — verification ONLY:
  pure, thread-safe, has a timeout, times itself.
- `compute_reward(verdict) -> float` — reward shaping, separate from
  verification, range [0,1].
- `estimate_cost()` — cold-start prior, never a measured cost.
- `metadata()` — dataset version, split sizes, manifest hashes, parameters.
- `split_ids(split)` — used by leakage tests.

**Why verification and reward are separate methods:** the noisy environment
is exactly the demonstration. Its verification is honest; only its reward is
corrupted. If shaping and verification were one call, we could not prove the
noise lives where we claim it does.

**The timeout** (`run_with_timeout`) uses a daemon worker thread, which
works identically on Windows and POSIX (Roadmap O.1 invariant 3 — no
`SIGALRM`). Phase 2 verifiers are bounded by design, so the timeout is a
safety net; unbounded loops belong to the Phase-11 code sandbox (stretch).

### 2.3 `envs/splits.py` — split assignment (Roadmap D.3, SPEC §3)

Four splits: `train`, `calib` (drives the scheduler online), `dev`
(tuning/validation curves), `test` (sealed, used once).

Assignment follows the frozen formula **literally**:

```
u = int(sha256(salt + dataset + problem_id)[:8], 16) / 2**32
```

The train pool is ranked by `u`; the first `calib_size` positions become
calib, the next `dev_size` become dev, the rest train. Ranking (instead of a
probability threshold) keeps the split sizes exact, is fully deterministic,
and changes completely under a different salt — which is why the salt is a
frozen config key (D-20: `curator-gate0-2026-10-d20`).

**Manifests:** `data/manifests/{env}_{split}.txt` (one id per line, sorted)
plus `MANIFEST.sha256` recording each file's SHA-256. Manifests are
*committed to Git* while data itself is gitignored — this makes the split
reviewable in a PR and makes any split change visible in the diff.
`verify_manifests()` fails loudly on tampering, which is what makes the
leakage tests meaningful.

**Sealed ids are in the manifest too** (`gsm8k_test.txt`): knowing the ids
of a sealed set leaks nothing, and the leakage tests need them to prove
disjointness.

### 2.4 The three environments

**GSM8K (`envs/gsm8k.py`)** — the familiar public benchmark; teaches nothing
about allocation by itself but is the reference arm.
- Official test (1,319 items) is written to `data/test_sealed/` and never
  served by the environment — only through the guard.
- The 7,473-item train pool is split 6,873 / 300 / 300.
- The prompt template asks for the final answer in `\boxed{}`; the verifier
  extracts the LAST boxed value, normalises numerics (commas, `$`, `%`,
  trailing zeros) and compares with **exact rational arithmetic**
  (`Fraction`), so `2.50 == 2.5 == 5/2`.
- Gold answers verify at 100% on real data; a wrong answer sets
  `parse_ok=True, success=False`; garbage sets `parse_ok=False`.

**Countdown (`envs/countdown.py`)** — the contamination-free procedural arm
(pretraining cannot have memorised instances we generate at runtime).
- Every prompt is fresh: the instance seed is drawn from a **split-specific
  disjoint range** (D-22: train [0, 1e7), calib [1e7, 1.5e7), dev
  [1.5e7, 2e7), test [2e7, 3e7)). Disjoint seed ranges are the Roadmap's
  own leakage protection for procedural environments (D.3 item 3).
- Targets are guaranteed solvable: the generator enumerates all exactly
  reachable values (recursive partition over index bitmasks, cached) and
  picks one in the configured target range.
- **`safe_arith_eval` is the safety-critical piece.** It NEVER calls `eval`:
  it parses with `ast`, applies a strict node whitelist (only `+ - * /`,
  integer literals, unary +/-), rejects names/calls/float literals/anything
  else, bounds expression length and nesting, checks the used numbers against
  the available multiset (each at most once; requiring all numbers is a
  config flag), and evaluates exactly with `Fraction`. Division by zero is a
  verifier failure, not an escape.
- Verification is honest: an expression that reaches the target with all
  numbers used exactly once scores 1.

**Noisy-reward (`envs/noisy.py`)** — the sanity arm.
- Serves the GSM8K train rows verbatim; verification is the honest GSM8K
  check; the true outcome is logged in `verdict.info
  ["diagnostic_true_success"]` for diagnosis only (the Signal Engine must
  never read it).
- The noise lives ONLY in `compute_reward`, two modes (config):
  - `random`: reward ~ Bernoulli(q = 0.35), *statistically independent of
    the outcome*. The draw is keyed by a SHA-256 hash of
    (prompt_id, response), so it is reproducible, while its marginal rate
    stays q for **every** response — independence by construction, not by
    approximation.
  - `flip`: the true reward flips with probability `flip_p`.
- Why per-(prompt, response) keys: a prompt-level key would give all G
  rollouts of a prompt the same reward, collapsing GRPO's within-group
  advantage signal. Real rollouts differ in text, so each gets its own draw.
- This environment is what the calibration mechanism must eventually flag
  (status S5, Gate 6): if calibration cannot catch "proxy rewards noise",
  the project's novelty claim fails.

### 2.5 `evaluation/guard.py` — the sealed-test guard

The sealed test is the only unbiased estimate of generalisation, so it gets
a hard, test-enforced gate. `SealedTest.open(...)` refuses unless:

1. `CURATOR_FINAL_EVAL=1` is set in the environment (nobody opens the test
   by accident),
2. the split is exactly `"test"`,
3. a frozen-config hash file exists (Gate 10 freeze rule) **and matches the
   current config hash** — no evaluation after sneaky reconfiguration,
4. this (checkpoint_hash, split) pair has never been opened — a second open
   raises.

On success it appends one JSONL entry to
`reports/test_access_log.jsonl` (itself gitignored) and serves
`data/test_sealed/<env>.jsonl` only through `load_items()`. Every refusal
path has a unit test. The production caller arrives in Phase 15; the guard
exists now so the discipline is testable before anything expensive runs.

### 2.6 Scripts and configuration

- `scripts/download_data.py` — pulls the official `openai/gsm8k` parquet
  files directly (no `datasets` dependency; pandas + pyarrow are already
  core deps). Records each file's SHA-256, the dataset revision id
  (`740312a...`), and the retrieval date in `data/raw/gsm8k/metadata.json`
  — the "[verify] record the exact revision" requirement of Roadmap D.2.
  Idempotent: matching files are not re-downloaded.
- `scripts/build_splits.py` — official test → sealed dir; train pool →
  hash-split via `assign_splits_finite`; processed JSONL to
  `data/processed/{env}/{split}.jsonl`; manifests written and indexed.
  **Byte-identical on re-run** (checked by hashing `MANIFEST.sha256` twice).
  Also derives the noisy environment's rows and enforces four-way
  disjointness before writing anything.
- `scripts/env_profile.py` — generates `reports/env_profiles/
  phase2_profile.json` (Roadmap D.5 item 6): prompt-generation rate and
  mean verifier seconds per environment, gold and garbage.
- `configs/base.yaml` — new `data.envs.{gsm8k,countdown,noisy}` sections
  (token limits, verifier timeout, countdown generator ranges, noisy
  mode/q/flip_p, cold-start priors) and the frozen salt. All scientific
  parameters have **no Python defaults** (Rule 5): a missing key fails
  config validation. The keys are listed in SPEC §7 so the spec-lint test
  keeps base.yaml and the spec in lockstep.

---

## 3. How it was verified (Gate 1 evidence)

Full suite: **71 tests pass, ruff clean, CPU-only** (Roadmap O.1 invariant
1 — no GPU anywhere in this layer).

| Check (Roadmap D.5) | Result |
|---|---|
| Determinism: same rng state → identical prompt | passes for all 3 environments |
| Gold answers verify ≥ 99% | 100% on 60–200-prompt samples per environment |
| Garbage / empty strings fail ≥ 99% | 100% |
| Split pairwise disjointness (all 4 splits) | enforced in `build_splits` + tested |
| Manifest reproducibility | two runs → identical `MANIFEST.sha256` |
| Sealed split never served by environments | `ValueError` on `generate_prompt("test")` for finite envs; countdown 'test' instances come only from the sealed seed range |
| Noisy success rate within 0.02 of design | |rate − 0.35| ≤ 0.02 for gold AND garbage responses (independence of outcome) |
| Bounded evaluation | oversized/nested/banned expressions rejected; verifier < 1 ms |
| Manifest tampering detection | hash mismatch and missing-file cases both detected |
| Guard refusal paths | all four refusal modes + once-only rule unit-tested |

Measured profiles (`reports/env_profiles/phase2_profile.json`):

| Env | Prompt generation | Verifier (gold) |
|---|---|---|
| countdown | ~64/s (target search) | ~0.9 ms |
| gsm8k | ~58k/s | ~0.6 ms |
| noisy | ~83k/s | ~0.3 ms |

---

## 4. Documented deviations from the Roadmap

| ID | Deviation | Why |
|---|---|---|
| D-20 | Split salt frozen now (`curator-gate0-2026-10-d20`) | The salt must exist before splits are built; manifests are committed regardless |
| D-21 | Env parameters live in `configs/base.yaml` (`data.envs.*`), not `configs/env/*.yaml` | Everything is inside the pydantic-validated, hashed `RootConfig`; separate files would duplicate keys and drift |
| D-22 | Countdown calib/dev seed-range boundary at 1.5e7 | Roadmap fixes 1e7/2e7 but leaves the calib/dev boundary open |

---

## 5. How to run everything

```bash
# 1. environment + dev tools
pip install -e ".[dev]"

# 2. data (one-time; idempotent after that)
python scripts/download_data.py    # GSM8K parquet -> data/raw/gsm8k/ + provenance
python scripts/build_splits.py     # processed JSONL + committed manifests

# 3. verification
python -m pytest                   # 71 tests: unit + contract + architecture
python -m ruff check src tests scripts
python scripts/env_profile.py      # regenerate reports/env_profiles/phase2_profile.json
```

Reproducibility notes:
- deleting `data/processed/` and `data/test_sealed/` and re-running
  `build_splits.py` reproduces the exact same splits (the salt is frozen
  and the formula is deterministic);
- the manifests in Git are the authority — if your local split differs,
  `verify_manifests` will say so.

---

## 6. What Phase 3 is, and why it comes next

Next is the **Curator simulator**: a synthetic world with known dynamics
(true skills, transfer matrix, costs), oracles that know the true optimum,
and a scenario suite — all producing the *same* `RoundObservation` schema
the real trainer will emit. The scheduler (Phases 4–5) is developed and
tuned there, because the simulator is the only place the correct answer is
known (Roadmap Decision 2: no GPU work before Gate 2 passes). Phase 2's
interface freeze is what makes that possible: the simulator can imitate the
real environments exactly because it only has to produce `Prompt`-free
`RoundObservation` streams through the same types.
