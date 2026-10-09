# Phase B pilot (Kaggle, 2026-10-09)

Measured with `scripts/kaggle_pilot.py` on one Tesla T4, fp16, plain HF generation, temperature 1.0, top_p 1.0, G = 8 samples per prompt, 64 prompts per environment, max_new_tokens 512, dev split (noisy: train split, train-only by design). Warm-up excluded; wall-clock timed with the GPU synchronised.

## Qwen2.5-0.5B-Instruct (peak GPU memory 1.83 GB)

| env | pass@1 | pass@8 | mixed groups | mean abs adv | parse ok | tokens | trunc | s/prompt (gen+verify) | rel. cost | batch CV |
|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 0.199 | 0.641 | 0.641 | 0.496 | 0.631 | 347 | 0.143 | 3.441 | 1.00x | 0.04 |
| countdown | 0.000 | 0.000 | 0.000 | 0.000 | 0.131 | 348 | 0.449 | 3.227 | 0.94x | 0.00 |
| noisy | 0.377 | 1.000 | 1.000 | 0.844 | 0.660 | 322 | 0.096 | 3.547 | 1.03x | 0.03 |

## Qwen2.5-1.5B-Instruct (peak GPU memory 4.77 GB)

| env | pass@1 | pass@8 | mixed groups | mean abs adv | parse ok | tokens | trunc | s/prompt (gen+verify) | rel. cost | batch CV |
|---|---|---|---|---|---|---|---|---|---|---|
| gsm8k | 0.406 | 0.891 | 0.875 | 0.696 | 0.814 | 276 | 0.041 | 6.353 | 1.00x | 0.06 |
| countdown | 0.000 | 0.000 | 0.000 | 0.000 | 0.109 | 78 | 0.029 | 6.085 | 0.96x | 0.01 |
| noisy | 0.354 | 1.000 | 1.000 | 0.834 | 0.840 | 244 | 0.023 | 6.667 | 1.05x | 0.07 |

## Reading

- **Cost is measurable:** batch CV 0.00–0.07, under the 10% target (Roadmap H.7) on shared Kaggle T4s. The wall-clock cost model is usable.
- **No cost spread:** the three built environments cost within ±6% of each other (0.94–1.05×) at both model sizes. The cost lever has nothing to act on until environments with different completion lengths or verifiers exist (MATH, MBPP).
- **1.5B costs 1.85× more per prompt** (6.35 vs 3.44 s) and is far from memory limits at either size.
- **GSM8K is learnable for 0.5B** (pass@1 0.20, pass@8 0.64, 64% mixed groups). Parse rate is only 63%, so much of the failure is output format (`\boxed{}`), which GRPO typically learns fast.
- **Countdown is zero for both models** (pass@8 = 0, parse 11–13%): it meets the too-hard criterion (pass@16 ≤ 0.02) and is the zero-signal arm for now.
- **Noisy works as designed:** pass@1 0.377 and 0.354 against the design value 0.35 (small-sample noise at 64 prompts).
