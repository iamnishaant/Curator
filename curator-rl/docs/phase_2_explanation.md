# Phase 2 — What We Built and Why (Explanation)

Phase 2 turns the Phase 1 skeleton into the data layer: a frozen environment
interface, the first three environments, deterministic split machinery with
committed manifests, and the sealed-test guard. This is everything Gate 1
checks.

---

## 1. What was built

```
src/curator_rl/core/types.py        Prompt, Verdict, RolloutGroup, CostPrior (Roadmap B.3)
src/curator_rl/envs/base.py         Environment ABC + cross-platform verifier timeout
src/curator_rl/envs/splits.py       hash-split assignment + manifests (Roadmap D.3)
src/curator_rl/envs/registry.py     name -> environment factories from data.envs config
src/curator_rl/envs/gsm8k.py        Gsm8kEnv (\boxed{} verifier, exact rational compare)
src/curator_rl/envs/countdown.py    CountdownEnv + safe_arith_eval (AST evaluator, no eval)
src/curator_rl/envs/noisy.py        NoisyRewardEnv (honest verifier, corrupted reward)
src/curator_rl/evaluation/guard.py  SealedTest: env-var + frozen-hash + access-log + once-only
scripts/download_data.py            GSM8K parquet download with SHA-256 + revision recorded
scripts/build_splits.py             official test -> sealed; train pool -> calib/dev/train
scripts/env_profile.py              per-env throughput + verifier timing (D.5 item 6)
tests/unit/test_splits.py           hash determinism, sizes, salt sensitivity, manifests
tests/unit/test_guard.py            all refusal paths, access log, second-open refusal
tests/contract/test_env_contract.py the Gate 1 suite (determinism, gold/garbage, sealing, noise rate)
```

## 2. The three environments

- **GSM8K** (finite): official test set is sealed to `data/test_sealed/`;
  the 7,473-item train pool is hash-split into train (6,873) / calib (300) /
  dev (300). The model must answer in `\boxed{}`; answers are compared with
  exact rational arithmetic after normalising commas/`$`/trailing zeros.
- **Countdown** (procedural): every prompt is a fresh instance; the instance
  seed comes from a split-specific disjoint range (D-22), so leakage is
  impossible by construction. Verification uses `safe_arith_eval` — a strict
  AST whitelist (only `+ - * /`, integer literals, unary +/-), bounded
  expression size, each number at most once (config can require all), exact
  `Fraction` arithmetic. There is no `eval` anywhere on this path.
- **Noisy-reward** (sanity arm): wraps the GSM8K train rows. Verification is
  the honest GSM8K check; noise lives ONLY in `compute_reward`. `random`
  mode draws reward ~ Bernoulli(q = 0.35) keyed by a hash of
  (prompt, response) — reproducible, and its marginal rate is q for every
  response, so the reward is statistically independent of the true outcome.
  The true success is logged in `verdict.info["diagnostic_true_success"]`
  for diagnosis only.

## 3. Splits and manifests (Roadmap D.3)

Assignment follows the frozen SPEC s3 rule literally:
`u = int(sha256(salt + dataset + problem_id)[:8], 16) / 2**32`. The train
pool is ranked by u; the first 300 items are calib, the next 300 are dev,
the rest train — exact sizes, deterministic, and different under a different
salt. Committed artifacts live in `data/manifests/`
(`{env}_{split}.txt` + `MANIFEST.sha256`); `verify_manifests` fails loudly
if a manifest file changes, which is what makes the leakage tests meaningful
in CI.

The sealed test is reachable ONLY through `evaluation.guard.SealedTest.open`,
which requires `CURATOR_FINAL_EVAL=1`, a frozen-config hash that matches the
current config, writes one JSONL access entry, and refuses a second open for
the same (checkpoint, split) pair. The unit tests exercise every refusal
path.

## 4. Verification results (Gate 1 evidence)

- Full suite: 71 tests pass CPU-only; ruff clean.
- Gold answers verify at 100% on 200-prompt samples per environment
  (contract suite checks >= 99%); garbage/empty fail 100%.
- Noisy reward rate within design value: measured |rate - 0.35| <= 0.02 for
  both gold and garbage responses (independence of outcome).
- Split manifests: two `build_splits.py` runs produce the identical
  `MANIFEST.sha256` (byte-identical, checked); the four splits are pairwise
  disjoint; official test = 1,319 sealed ids.
- Countdown seed ranges disjoint by test; oversized/banned expressions
  rejected; division by zero is a verifier failure, not an exception escape.
- `reports/env_profiles/phase2_profile.json`: generation ~60-80/s for
  countdown (target search) and >50k/s for the finite envs; verifier cost
  < 1 ms per response in all three environments.

## 5. Documented deviations (see DECISIONS.md)

- **D-21** — environment parameters live in `configs/base.yaml`
  (`data.envs.*`) instead of `configs/env/*.yaml`, so they are inside the
  config hash and validated by the pydantic loader.
- **D-22** — Countdown calib/dev seed-range boundary fixed at 1.5e7.
- **D-20** — split salt frozen at `curator-gate0-2026-10-d20`.

## 6. What comes next (Phase 3)

The Curator simulator: a synthetic world with known dynamics, oracles, and
the same `RoundObservation` schema — the testbed the scheduler (Phase 4-5)
is developed against before any GPU work.
