# Phase E pilot (Kaggle, 2026-10-09): all five environments, corrected sampling

Measured with `scripts/kaggle_pilot.py` on Tesla T4, fp16, plain HF generation, temperature 1.0, top_p 1.0,
**top_k disabled and repetition_penalty 1.0** (D-84 fix), G = 8, 64 dev prompts per environment
(MBPP: its 60 dev prompts; noisy: train split, D-21), `max_new_tokens` 512 for every environment
(note: MATH35's own cap is 1024, so its pass rates here are a lower bound).

## Qwen2.5-0.5B-Instruct (peak GPU memory 2.05 GB)

| env | pass@1 | pass@8 | mixed groups | mean abs adv | parse ok | tokens | trunc | s/prompt (gen+verify) | rel. cost | batch CV |
|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 0.344 | 0.688 | 0.672 | 0.536 | 0.750 | 334 | 0.109 | 3.334 | 1.00x | 0.01 |
| math35 | 0.117 | 0.344 | 0.344 | 0.252 | 0.373 | 464 | 0.613 | 3.569 | 1.07x | 0.12 |
| mbpp | 0.183 | 0.500 | 0.483 | 0.368 | 0.992 | 136 | 0.013 | 4.414 | 1.32x | 0.16 |
| countdown | 0.000 | 0.000 | 0.000 | 0.000 | 0.148 | 300 | 0.420 | 3.037 | 0.91x | 0.00 |
| noisy | 0.344 | 0.984 | 0.984 | 0.820 | 0.768 | 299 | 0.043 | 3.357 | 1.01x | 0.03 |

## Qwen2.5-1.5B-Instruct (peak GPU memory 5.12 GB)

| env | pass@1 | pass@8 | mixed groups | mean abs adv | parse ok | tokens | trunc | s/prompt (gen+verify) | rel. cost | batch CV |
|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 0.559 | 0.922 | 0.812 | 0.660 | 0.801 | 254 | 0.021 | 5.723 | 1.00x | 0.12 |
| math35 | 0.258 | 0.500 | 0.484 | 0.361 | 0.414 | 447 | 0.559 | 6.731 | 1.18x | 0.11 |
| mbpp | 0.423 | 0.733 | 0.650 | 0.485 | 1.000 | 125 | 0.004 | 6.471 | 1.13x | 0.18 |
| countdown | 0.000 | 0.000 | 0.000 | 0.000 | 0.242 | 63 | 0.045 | 5.736 | 1.00x | 0.01 |
| noisy | 0.332 | 0.953 | 0.953 | 0.802 | 0.854 | 228 | 0.006 | 5.325 | 0.93x | 0.17 |

## Portfolio acceptance rule (Roadmap v3 §4.1), Qwen2.5-0.5B

| Criterion | Result | Verdict |
|---|---|---|
| >= 2 arms with base pass@8 in [0.15, 0.85] | gsm8k 0.688, math35 0.344, mbpp 0.500 (3 arms) | **pass** |
| too-hard arm with pass@16 <= 0.02 | countdown: 0 successes in 512 samples (64 prompts x 8); pass@16 not measured directly | consistent, not proven |
| measured cost spread (max/min unit cost) >= 2x | 1.45x under HF generation (mbpp 1.32x / countdown 0.91x) | **not met under HF; re-measure under vLLM** |

The escalation rule (switch to 1.5B if fewer than 2 arms qualify) is **not triggered**: Level 1 stays on 0.5B.
1.5B costs 1.7x more per prompt (5.72 vs 3.33 s on GSM8K).

## Reading

- **The D-84 sampling diagnosis is confirmed.** GSM8K pass@1 rose from 0.199 to 0.344 (pass@8 0.641 to 0.688,
  parse rate 0.63 to 0.75) once Qwen's `top_k=20` and `repetition_penalty=1.05` were no longer inherited.
- **MATH35 and MBPP are learnable for 0.5B.** MATH35 pass@8 0.34 with 61% truncation at 512 tokens: the cap,
  not only skill, limits it; training uses its own cap of 1024. MBPP pass@8 0.50 with a 99% parse rate.
- **The cost spread is hidden by the measurement method.** HF `generate` runs a batch until its longest
  completion ends, so a batch costs about the same whatever the typical answer length. Completion tokens
  differ 3.4x (MATH35 464 vs MBPP 136) but measured cost only 1.07x vs 1.32x; MBPP's extra cost is its
  verifier (~0.11 s per completion, run serially here). vLLM, which training uses, schedules sequences
  continuously, so its cost follows tokens. `scripts/kaggle_cost_probe.py` measures this (D-86).
- **Batch CV of 0.11-0.18** on some environments exceeds the 10% target (Roadmap H.7), but it compares
  batches of *different* prompts, so it mixes content variation with measurement noise. Repeatability must
  be judged on identical batches (the drift monitor, H.7), not from this column.
- **Countdown stays the zero-signal arm** for both models (parse 15-24%).
- **Noisy works as designed** (pass@1 0.344 / 0.332 vs design value 0.35).
- **TRL GRPO smoke test (HF backend) failed on an environment issue**: Kaggle ships `torchao` 0.10 and PEFT
  0.20 refuses versions below 0.16 when injecting LoRA. Fix: `pip uninstall -y torchao` before installing TRL
  (notebooks patched). The vLLM venv run (D-83) was unaffected.
