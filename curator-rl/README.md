# curator-rl

Cost-aware environment selection for agentic RL training of LLMs — the
scheduling layer around an unmodified GRPO trainer.

- **Spec:** [`docs/SPEC.md`](docs/SPEC.md) (frozen contract; the MVP/Stretch
  scope table and notation live there)
- **Decisions log:** [`docs/DECISIONS.md`](docs/DECISIONS.md)
- **Phase 1 in detail:** [`docs/phase_1_explanation.md`](docs/phase_1_explanation.md)
  (what was built, why, and how it verified)
- **Phase 2 in detail:** [`docs/phase_2_explanation.md`](docs/phase_2_explanation.md)
  and [`PHASE2_README.md`](PHASE2_README.md) (what was built and why)
- **Phase 3 in detail:** [`PHASE3_README.md`](PHASE3_README.md)
  (comprehensive: what, how, and why) and the shorter
  [`docs/phase_3_explanation.md`](docs/phase_3_explanation.md)
- **Roadmap:** see `CURATOR_Implementation_Roadmap.md` v2.0 in the project
  folder (Phases 0–16, gates, experiment protocol)

## Status

Phase 3: Curator simulator (Roadmap Part F) — synthetic world with known
dynamics, DP/static/myopic oracles, the S-A..S-H scenario suite, the episode
harness with the budget-ledger stop rule, and `experiments/run_sim.py`.
Every scenario discriminates: the oracle beats Uniform by > 10%.

## Quickstart (Phase 1)

```bash
pip install -e ".[dev]"

make lint        # ruff check
make test        # pytest
make smoke       # init-run on configs/experiment/smoke.yaml
```

## Quickstart (Phase 2 data)

```bash
python scripts/download_data.py    # GSM8K parquet -> data/raw/gsm8k/
python scripts/build_splits.py     # hash-split -> processed JSONL + manifests
python scripts/env_profile.py      # reports/env_profiles/phase2_profile.json
```

`init-run` creates `runs/<run_id>/` with a frozen `config.yaml` and
`metadata.json`, and prints one JSON line
(`run_id`, `run_dir`, `config_hash`). It is idempotent for identical configs
and refuses a conflicting config for the same run id.

## Quickstart (Phase 3 simulator)

```bash
python experiments/run_sim.py --scenario configs/sim/scenario_sa.yaml \
    --methods uniform,random,static_oracle,myopic_oracle,dp_oracle --seeds 20
```

Runs any method set on a scenario and writes a JSON report (final scores,
oracle-gap closure) to `reports/sim/`. Scenarios: `configs/sim/scenario_s?.yaml`
(S-A..S-H). The DP oracle only supports N ≤ 3 worlds without drift/late-start.

## Platform (Roadmap v2.0 O.1)

- Develop on Windows + WSL2 (or Linux).
- GPU runs **only** on Linux (WSL2 with CUDA, Colab, or a university server).
- The simulator and all scheduler logic run CPU-only — a GPU dependency in
  L1/simulator fails the import-rule test.
