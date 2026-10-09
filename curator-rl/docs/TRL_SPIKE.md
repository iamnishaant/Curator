# TRL spike (Roadmap v2 G.2, v3 Phase B) — results of the first probe

Run with `scripts/kaggle_trl_probe.py` on Kaggle, one Tesla T4, 3 LoRA GRPO steps, Qwen2.5-0.5B-Instruct, fp16.

## Versions that worked (pin these)

python 3.13.15 · torch 2.11.0+cu128 · transformers 5.16.1 · trl 1.15.0 · peft 0.20.0 · datasets 4.8.5 · accelerate 1.14.0. vLLM and bitsandbytes were not installed. Install with `pip install trl==1.15.0` in the notebook so a changed Kaggle image cannot move the version silently.

## Answers to the G.2 questions

| # | Question | Finding |
|---|---|---|
| 1 | Group contiguity | With a `Dataset`, the 8 completions of a prompt arrive **contiguously** (`prompt_id` run of 8). An `IterableDataset` was not tested. |
| 2 | Where to time | `step_time` is in `trainer.state.log_history`; a `TrainerCallback` with `on_step_begin/end` plus `torch.cuda.synchronize()` gave step timings (22.1, 16.9, 17.0 s). No private methods needed. |
| 3 | Token counts | `completion_ids` **is passed** to the reward function (padding behaviour not yet checked). |
| 4 | Dataloader lag | Not tested yet (needs the dynamic prompt stream). |
| 5 | LoRA + fp16 | Works on T4; peak GPU memory 1.55 GB at 256 completion tokens. |

## Facts the adapter depends on

- Reward function receives: `prompts`, `completions` (a list of message lists, each `[{"role","content"}]`), `completion_ids`, every extra dataset column (`env_id`, `prompt_id`, `reference`, `meta_json` arrived as per-completion lists), plus `trainer_state`, `log_extra`, `log_metric`.
- `GRPOTrainer.__init__` accepts `callbacks`, `peft_config`, `rollout_func`, `tools` and `environment_factory` (the last two are relevant to a later tool-use environment).
- **Defaults to set deliberately:** `loss_type='dapo'`, `scale_rewards='group'`, `beta=0.0` (no KL term), `epsilon=0.2`, token-level importance sampling, `temperature=1.0`, `num_generations=8`, `use_vllm=False` (colocate mode available, `vllm_gpu_memory_utilization=0.3`). Record the chosen values in the frozen protocol.
- **Group-advantage convention:** `scale_rewards='group'` matches the unbiased-std advantage used by `core/advantage.py`.

## Problems the probe exposed

- **256-token completions are far too short for GSM8K.** The first step had `clipped_ratio = 1.0` (every completion truncated, reward 0, zero gradient). Real runs need at least 512 tokens (pilot: GSM8K mean 347 tokens, 14% truncation at 512 for 0.5B).
- **Zero-signal groups are common early** (`frac_reward_zero_std = 1` on 2 of 3 steps at pass@1 ≈ 0.1), consistent with the pilot's pass rates; dynamic filtering or easier prompts matter early in training.
- **Step time is long with plain HF generation:** about 17 s for 8 completions of ≤ 256 tokens (one prompt per step). Decoding latency, not batch size, dominates, so a step with P = 16 prompts × G = 8 at 512 tokens is estimated at roughly 45–90 s. This is an estimate, not a measurement; the next probe measures it directly.

## Not yet answered

IterableDataset behaviour, weight-update lag, vLLM colocate on T4 (not installed), padding in `completion_ids`, and the step time at the real batch shape.

## Step-time and vLLM probe (second Kaggle run)

- **vLLM colocate works on a Tesla T4** (compute capability 7.5). It must be installed in a separate environment: `python -m venv` fails on Kaggle's Python 3.13 image (`ensurepip` / `wrapt` error), `pip install virtualenv` then `python -m virtualenv` works. vLLM logged that FlashAttention 2 is unsupported below capability 8 and fell back automatically; CUDA graphs captured.
- **Measured at the real shape** (P = 16 prompts × G = 8, 512 max tokens, LoRA, fp16, GSM8K): **42.2 s per optimizer step** after warm-up (37.1, 37.7, 40.8, 40.2 s logged), 128 completions per step, mean completion length 296–321 tokens, peak GPU memory 6.11 GB, vLLM `gpu_memory_utilization 0.3`.
- **Training signal at 512 tokens is healthy:** truncation 3–11% (versus 100% at 256 tokens in the first probe), mean reward 0.25–0.51, zero-variance groups 6–38% (about 20% on average), so about 80% of groups carry gradient.
- Not yet known: the HF-backend step time at the same shape (the vLLM cell was the only output received), and the generation/update split. If the update dominates, vLLM saves little.
