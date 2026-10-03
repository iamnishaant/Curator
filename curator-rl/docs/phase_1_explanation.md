# Phase 1 — What We Built and Why (Explanation)

This document explains the **Phase 1 milestone** of the CURATOR project in plain
language: what was implemented, why each piece exists, how the pieces work
together, and what was verified. It complements (not replaces) `docs/SPEC.md`
(the formal contract) and the v2.0 roadmap.

CURATOR = a cost-aware scheduler that decides, during LLM RL training, how to
split a fixed GPU budget across multiple training environments. Phase 1 does
**not** contain the scheduler or any training logic — it is the tested
skeleton that guarantees every later phase is reproducible, configurable, and
un-breakable by accident.

---

## 1. What CURATOR is, in one paragraph

An LLM agent is trained with GRPO (a reinforcement-learning post-training
method) across several environments — math, code, logic, tool use. Each
environment consumes GPU compute for different costs. Training too long on an
environment the model has already mastered wastes compute; training on one
that is too hard produces no learning signal at all. CURATOR treats
environment selection as a **bandit problem** (Discounted UCB over
environments), scores each environment with a cheap "proxy reward"
(learning progress + signal richness, normalised by measured cost), and
periodically re-calibrates that proxy against real held-out benchmark gains.
At the end it produces a validated ROI leaderboard of which environments were
worth their compute, checked via leave-one-out runs.

Phase 1 contains none of that logic yet. It contains exactly the
infrastructure the roadmap demands **before** any scientific code.

---

## 2. Repository created this phase

Location: `F:\sem 7\RL_project\curator-rl\`

```
curator-rl/
├── docs/
│   ├── SPEC.md                     frozen specification + MVP scope + hyperparameter list
│   ├── DECISIONS.md                decision log (defects D-1..D-15 pre-loaded)
│   └── phase_1_explanation.md      this document
├── configs/
│   ├── base.yaml                   every scientific hyperparameter (roadmap E.7)
│   └── experiment/smoke.yaml       inherits base via `include: ../base.yaml`
├── src/curator_rl/
│   ├── __init__.py                 package version
│   ├── cli.py                      init-run | show-config | env-info
│   ├── core/
│   │   ├── config.py               pydantic models, include-merge, overrides, config_hash
│   │   ├── seeding.py              named, order-independent RNG streams
│   │   ├── runmeta.py              git/platform/hardware metadata, run IDs
│   │   ├── jsonl.py                append-only JSONL logs with schema version
│   │   ├── atomic.py               crash-safe file writes (temp + fsync + rename)
│   │   └── paths.py                run directory layout
│   ├── envs/ signals/ scheduler/ cost/ calibration/ roi/ trainer/ simulator/ evaluation/ viz/
│       (empty packages — placeholders so the architecture test has real targets)
├── tests/
│   ├── conftest.py                 shared fixtures (base config loader, config factory)
│   ├── unit/                       test_config, test_seeding, test_runmeta, test_jsonl,
│   │                               test_atomic, test_paths
│   ├── integration/test_init_run.py
│   └── architecture/test_import_rules.py
├── pyproject.toml                  package metadata + dev/train/sim extras
├── Makefile                        install | test | lint | smoke
├── README.md
├── .gitignore                      never commits data/, runs/, checkpoints/, logs/
├── .pre-commit-config.yaml         ruff + yaml checks on every commit
└── .github/workflows/ci.yml        CPU-only lint+test on push/PR (Python 3.11 & 3.12)
```

---

## 3. The six core modules, explained

### 3.1 `core/config.py` — the validated configuration system

This is the most important file in Phase 1. Every experiment from now on is
described completely by a config file; no hyperparameter may ever live in
Python code.

What it enforces:

- **Pydantic models with `extra="forbid"`** — a typo in a YAML key is a hard
  error, not a silent default. This is what keeps two teammates from running
  "the same" experiment that is actually different.
- **Scientific hyperparameters have no Python defaults.** If `scheduler.gamma`
  is missing from the YAML, loading fails. The starting values live only in
  `configs/base.yaml` (the roadmap Rule 5: defaults are pilot-measured, never
  hard-coded).
- **Nested overrides** — `--set scheduler.tau=0.5 --set experiment.seed=3`
  patches any dotted path in the config, with YAML-typed values (`0.5` becomes
  a float, `true` becomes a bool). A bad path (e.g. `does.not.exist=1`) is rejected.
- **Include merging** — experiment configs like `smoke.yaml` inherit from
  `base.yaml` (`include: ../base.yaml`), deep-merged with child-wins semantics.
- **`config_hash(cfg)`** — SHA-256 of the canonical (sorted-key) JSON of the
  whole config. Two identical experiments hash identically regardless of key
  order; changing *any* hyperparameter changes the hash. This key is what
  makes experiment IDs and the frozen-config checks (later gates) possible.
- **Known names section:** the notation collision of the original Part 2
  (S-4: `c` meaning both cost and the UCB exploration coefficient) is resolved
  in code: the coefficient is `scheduler.exploration_coef`, cost is
  `unit_cost`, the bandit forgetting factor is `scheduler.gamma` (documented
  as *not* an MDP discount).

### 3.2 `core/seeding.py` — order-independent random streams (Roadmap Part N)

The scheduler samples prompts, the simulator draws worlds, the model
initialises LoRA weights — all need independent randomness that is exactly
reproducible **regardless of call order**.

- `SeedManager(master_seed).rng("data_order")` returns a NumPy `Generator`
  derived from `SeedSequence(entropy=master, spawn_key=hash(name))`. Asking for
  stream `a` then `b` gives exactly the same draws as asking `b` then `a`.
- `int_seed(name, bits=32)` derives a plain integer seed (for e.g. zipping a
  python `random` or an external tool) from `sha256(f"{master}:{name}")`.
- Unit tests prove: same seed → same draws; different names → uncorrelated
  draws; order independence.

### 3.3 `core/runmeta.py` — run identity

`collect_run_metadata()` snapshots the machine state for every run: git commit
SHA and dirty flag (a dirty tree will later be *refused* for final runs —
Roadmap Part N), a hash of the uncommitted diff, Python/platform versions,
package versions, GPU info from `nvidia-smi`, CPU count, UTC timestamp.

`make_run_id(cfg, meta)` composes the roadmap's format
`{date}_{tier}_{method}_s{seed}_{git7}_{cfg6}`, e.g.

```
20261002_0_uniform_s0_nogit_31652c
```

(the git segment is `nogit` until the repo is initialised). Two distinct files
freezing the same experiment on two machines produce the same ID prefix
structure — which is how runs become comparable across people.

### 3.4 `core/jsonl.py` — the experiment record

All per-round logging later (weights, signals, costs) goes to append-only
JSONL files, because a CSV cannot append across crash boundaries and a
database is too heavy.

- `JsonlWriter(path, schema_version, fsync)` stamps every record with
  `_schema` and a monotonically increasing `_seq`. **Reopening an existing file
  continues `_seq` from where it left off** — a crash never reuses a number.
- `read_jsonl(path, skip_corrupt_tail=True, on_skip=cb)` yields records and, if
  the process died mid-line, yields everything complete and reports the number
  of skipped lines through a callback (or raises if asked to be strict).
- Tests cover round-trip, append, `_seq` continuation, and the truncated-tail case.

### 3.5 `core/atomic.py` — crash-safe writes

`atomic_write_text` / `atomic_write_json` write to a temp file in the *same*
directory, flush + `fsync`, then `os.replace` (atomic on both POSIX and
Windows). If the process dies between write and rename, the old file survives
and no `.tmp` litter is left behind. The unit test **simulates a crash**
(monkey-patched `os.replace` raising OSError) and asserts exactly that. Later
this guarantees never corrupting `config.yaml`, `metadata.json`, checkpoints
or the scheduler state file mid-run.

### 3.6 `core/paths.py` — run layout

`RunPaths.create()` idempotently builds the roadmap's run directory:

```
runs/{run_id}/config.yaml      frozen copy of the config
runs/{run_id}/metadata.json    runmeta + config hash
runs/{run_id}/logs/            rounds.jsonl etc. (written from Phase 3 on)
runs/{run_id}/checkpoints/     model + scheduler state (Phase 7+)
runs/{run_id}/reports/         figures, tables, leaderboard exports
```

---

## 4. The CLI

`python -m curator_rl.cli` (or `curator`, once PATH includes the scripts dir):

| Command | What it does |
|---|---|
| `init-run --config <yaml> [--set k=v ...]` | Creates `runs/{run_id}/` with the frozen `config.yaml` + `metadata.json`, prints one JSON line `{"run_id", "run_dir", "config_hash"}`. **Idempotent**: same config → same run id → reuses the directory; **conflicting** config for an existing run id → hard error. |
| `show-config --config <yaml>` | Prints the fully-merged, validated config as YAML — the "what would I actually run?" check. |
| `env-info` | Prints the platform/git/hardware snapshot as JSON. |

---

## 5. `configs/base.yaml` — the hyperparameter surface

Every scientific hyperparameter named in `docs/SPEC.md` §7 is present with its
roadmap starting value, grouped by component:

- `scheduler.*` — gamma 0.95, exploration_coef 0.5, tau 1.0, epsilon 0.10,
  warmup 6 rounds, score_norm `zscore`, cost_exponent 1.0, status_control
  `soft`, pull_unit `round`.
- `signals.*` — pass-rate discount λ=0.9, fast/slow window discounts 0.7/0.95
  (the LP-A estimator), richness mode, Beta prior (1,1), all status-classifier
  thresholds (p_sat 0.85, p_hard 0.10, z-scores, hysteresis, dwell time).
- `proxy.alpha/beta` 0.5/0.5 (fallback until the first calibration fit).
- `calib.*` — K=5 rounds, domain-target regression, trust region 0.25, S5 mismatch thresholds.
- `cost.*`, `data.*` (split salt!), `budget.*`, `steps_per_round=5`,
  `prompts_per_step=16`, `group_size=8`.

Two deliberate placeholders: `data.split_salt: CHANGE_ME_AT_GATE_0` (chosen
once, frozen) and an empty `train` dependency extra (torch/TRL get pinned in
the Phase 7 version spike, not installed half-pinned now).

---

## 6. Architecture guard: `tests/architecture/test_import_rules.py`

The roadmap's layering rules (B.2) are enforced by *scanning the source with
Python's `ast` module* on every test run:

- `core` imports nothing from the project.
- L1 pure packages (`signals`, `scheduler`, `calibration`, `roi`) may import
  only `core` + numpy/scipy — never torch/transformers/trl/datasets and never
  a Layer-2 package. (Pure, thus testable on the simulator.)
- TRL may only ever be imported inside `trainer/trl_adapter.py` (single
  version-sensitive file, per strategic decision 8).
- Nothing outside `evaluation/` may import the sealed final-test loader.

Today the L1 packages are empty, so the test passes trivially — but it is the
safety net that will catch accidental coupling **the moment** Phase 2+ code
lands.

---

## 7. What was verified (the roadmap's success criteria)

| Criterion | Result |
|---|---|
| All tests pass on CPU-only clean checkout | **31 passed** |
| `ruff check` clean | **passed** |
| Removing any scientific hyperparameter makes `load_config` fail | tested (`test_missing_scientific_param_raises`, 8 invalid-value cases) |
| Unknown keys refused | tested |
| Same config + seed → identical `config_hash` / stable across machines | tested (64-hex SHA-256, order-insensitive) |
| Override syntax works; bad override raises | tested |
| Spec ↔ config correspondence (spec lint) | tested — every hyperparameter in SPEC.md exists in base.yaml |
| `init-run` idempotent for identical configs; conflict refused | tested (unit + integration) |
| JSONL append/continue, corrupt-tail tolerance | tested |
| Atomic write survives simulated crash | tested |
| Seeded streams reproducible, independent, order-stable | tested |
| Core coverage ≥ 90% | **94%** |

Actual commands (Windows, from `curator-rl/`):

```
python -m pytest tests                       -> 31 passed
python -m ruff check .                       -> All checks passed
python -m curator_rl.cli init-run --config configs/experiment/smoke.yaml --set experiment.seed=0
{"run_id": "20261002_0_uniform_s0_nogit_31652c", "run_dir": "runs\\20261002_0_uniform_s0_nogit_31652c", ...}
# re-running the same command prints the same line with "reused": true
```

---

## 8. Design decisions worth remembering (also in DECISIONS.md)

1. **Package named `curator_rl`, repo `curator-rl`** — avoids a collision with
   the unrelated existing `curator` pip package.
2. **Config hash = SHA-256 over canonical JSON** — and everything else
   (run IDs, frozen dirs, later gate checks) builds on it.
3. **`include:` semantics** in configs: children win, relative paths resolve
   against the including file.
4. **`lambda` is a Python keyword**, so the signal discount is the field `lam`
   with YAML alias `signals.lambda` — the YAML surface stays as documented.
5. **No `uv.lock` at Phase 1** (no `uv` in this environment). The roadmap's
   lock-file requirement is satisfied by pinning the dependency floor in
   `pyproject.toml` for now; a lock (requirements.lock or uv.lock) is added
   with the Phase 7 `train` extra, when the version spike pins torch/TRL.
6. **Sandbox/CPU-only invariant stood up early** — the import-rule test plus
   the CI workflow running with `-m "not gpu"` makes "simulator and scheduler
   must run GPU-less" a checked property, not an intention.

---

## 9. What was *not* done (deliberately)

- No scientific logic: no environments, signals, scheduler, cost meter, GRPO.
- No git repo / commit yet: the roadmap's Phase 1 commit + `phase-1` tag are
  ready to go but were left to the team (also: `runmeta` will start reporting
  real git SHAs only after `git init`).
- `data.split_salt` intentionally still `CHANGE_ME_AT_GATE_0` — Gate 0 freezes it.
- Part 2/proposal document edits (references S-15, dashboard/verl wording) are
  recorded in `DECISIONS.md` but must still be applied to the PDF itself.

---

## 10. Next milestone (Phase 2)

`core/types.py` data contracts (`Prompt`, `Verdict`, `RolloutGroup`), the
frozen environment interface + registry, the hash-based `SplitManager` +
`build_splits.py`, the sealed-test guard skeleton, and the first three
environments (GSM8K, Countdown with a safe AST evaluator, noisy-reward) with
their contract tests — ending at **Gate 1**. Nothing above changes: the Phase 2
work slots into the `envs/` layer the skeleton already reserves.
