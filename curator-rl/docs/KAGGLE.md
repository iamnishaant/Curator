# Running CURATOR on Kaggle (Phase B pilot)

This is the first GPU contact. It answers three questions that no amount of simulator
work can: **which model** (0.5B or 1.5B), **how much each environment really costs**, and
**what TRL's GRPOTrainer actually does** (so the trainer adapter is written against facts).
It does not train a useful model, and it never touches the sealed test set.

## One-time setup (about 10 minutes)

1. **Build the bundle on your PC** (from the `curator-rl` folder):

   ```powershell
   python scripts/make_kaggle_bundle.py
   ```

   This writes `dist/curator-rl-bundle.zip` (about 3 MB: `src`, `configs`, `scripts`,
   `experiments`, `data/manifests`, `data/processed`). The script refuses to include the
   sealed test set, raw data, checkpoints or caches (and a test enforces that).

2. **Upload it as a Kaggle Dataset.** kaggle.com → *Datasets* → *New Dataset* → drag in
   `curator-rl-bundle.zip` → name it `curator-rl-bundle` → *Create* (keep it **Private**).
   Whenever the code changes, rebuild the zip and upload a *new version* of the same dataset.

3. **Create the notebook.** kaggle.com → *Code* → *New Notebook* → *File → Import notebook* →
   choose `notebooks/kaggle_pilot.ipynb`.

4. **Notebook settings** (right sidebar):
   - **Accelerator: GPU T4 x2** (two GPUs let the 0.5B and 1.5B measurements run in parallel).
     Do **not** pick P100 (no vLLM support later, and slower fp16).
   - **Internet: On** (downloads the Qwen models and `trl`; phone verification may be needed).
   - **Add data**: attach your `curator-rl-bundle` dataset.
   - **Persistence**: "Files only" is enough.

## Run

Run the cells top to bottom (or *Run All*). What each does:

| Cell | Does | Time |
|---|---|---|
| 1 | `nvidia-smi`, library versions | seconds |
| 2 | unpacks the bundle to `/kaggle/working/curator-rl`, checks the sealed set is absent, loads the config | seconds |
| 3 | **the pilot**: Qwen2.5-0.5B on GPU 0 and Qwen2.5-1.5B on GPU 1, in parallel; 64 dev prompts × 8 samples per environment | roughly 10–25 min |
| 4 | prints the result tables | seconds |
| 5 | installs TRL and runs a 3-step LoRA GRPO smoke test (the "TRL spike") | roughly 5–10 min |
| 6 | prints `trl_probe.json` and the list of output files | seconds |

The whole notebook should use **well under 1 GPU-session-hour**, so it is cheap relative to
the weekly quota.

## What to send back

Paste into the chat (or attach):

1. The output of **cell 4** (the two markdown tables).
2. The printed **summary of cell 5** (stage ok/FAILED lines, step seconds, peak GPU GB), and, if a stage
   FAILED, the `error` and `traceback` fields from cell 6.
3. Anything odd in `/kaggle/working/pilot/log_*.txt`.

To keep the raw files: *Save Version → Save & Run All (Commit)*, then download from the
notebook's *Output* tab: `pilot/*.json`, `pilot/samples_*.jsonl` (completions for the manual
audit), `trl_probe/trl_probe.json`.

## How the numbers will be used

| You see | It means | Decision |
|---|---|---|
| `pass@1` of GSM8K/MATH-style arms between 0.15 and 0.85 | learnable arm | portfolio acceptance rule (v3 §4.1): need at least 2 such arms |
| Countdown `pass@G` ≈ 0 and mixed groups ≈ 0 | too hard for this model (small models often fail it) | it becomes the too-hard arm, or we switch to an easier variant or to 1.5B |
| `rel. cost` spread across environments | whether the cost lever has anything to act on | spread under 2× → add a long-completion or code arm sooner |
| `trunc` high (completions cut at 512 tokens) | the cap hides correct answers | raise the cap for that arm, which raises its cost |
| `batch CV` above 0.10 | wall-clock cost is too noisy on shared Kaggle machines | switch to the token-based cost model (`COST_MODEL.md`) |
| noisy `pass@1` ≈ 0.35 | the noisy environment works as designed | none |
| GRPO `step_seconds` and `peak_gpu_memory_gb` | real round length and memory headroom | fixes R, K, B by the horizon rule (v3 §4.2) |
| `reward args` / `contiguous` / `completion_ids passed` | the facts the trainer adapter depends on | `docs/TRL_SPIKE.md` |

## If something breaks

- **`bundle not found`** in cell 2: the dataset is not attached; add it in *Add data*.
- **CUDA out of memory**: the pilot halves `--batch-prompts` automatically; for the TRL probe
  lower `--group-size` or `--max-completion-length`.
- **Model download fails**: Internet is off, or Hugging Face is rate-limiting; retry.
- **Only one GPU visible**: run the two pilot commands one after the other instead of in parallel.
- **TRL probe stage FAILED**: expected on a first run. The traceback is the finding; send it
  and the probe will be adjusted (it has never been run against a real TRL install).
- **Session interrupted**: all outputs are in `/kaggle/working`; nothing here needs resuming.

## What this does not do yet

- No MATH, MBPP or tool-use environments exist yet (Phase E); the pilot covers GSM8K,
  Countdown and the noisy arm.
- No training of consequence: that starts after the pilot fixes the model and the horizon.
- The 1.5B measurement is for the escalation decision only (v3 §4.6).
