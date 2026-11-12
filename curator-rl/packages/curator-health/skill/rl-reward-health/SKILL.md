---
name: rl-reward-health
description: Check whether the reward functions in an RL-with-verifiable-rewards (RLVR / GRPO / TRL) training setup actually mean anything, before and during training, using the curator-health package. Use when the user is training an LLM with GRPO/RLOO/PPO on programmatic rewards (math checkers, unit tests, schema validators), mixes several task environments, asks "is my reward broken / noisy / hackable", sees reward rising without eval gains, or wants to pick an environment mix.
---

# RL reward health (curator-health)

Two label-free, zero-GPU checks on one question: **does this reward depend on the prompt at all?**

1. **Pre-flight probe** (before training): score junk answers (empty, echoed prompt, empty/wrong
   `\boxed{}`, number spray, code that exits early) through each environment's real reward path.
   Junk that earns reward = a verifier bug that RL will learn.
2. **In-run audit** (during training, free): real verifiers make group pass rates *over-dispersed*
   across prompts (`D = 3–4` measured on Qwen2.5 0.5B/1.5B). A reward that ignores the prompt has
   `D ≈ 1`. An arm is flagged when the upper confidence bound of `log D̂` is below `log d_max` (1.5).

It observes only. It never changes the reward the trainer sees, and every entry point fails open.

## Step 0 — install

```bash
pip install "git+https://github.com/AnshBajpai05/Curator-RL#subdirectory=curator-rl/packages/curator-health"
# or, from a local checkout:  pip install -e curator-rl/packages/curator-health
```
Stdlib only; safe to add to a training environment.

## Step 1 — find the user's setup (read, don't guess)

Locate: the reward function(s), the dataset per environment ("arm"), the group size `G`
(`num_generations` in TRL), and whether rows carry `env_id` / `prompt_id` columns. If several
environments share one reward function, ask which column distinguishes them.

## Step 2 — pre-flight probe (always do this first; seconds, no GPU)

The probe calls `verifier(prompt: ProbePrompt, response: str) -> float`. Write a tiny adapter module
around the user's reward if its signature differs (TRL rewards take `(prompts, completions, **cols)`
and return a list), e.g.:

```python
# probe_adapters.py
from my_rewards import gsm8k_reward
def gsm8k(prompt, response):
    return gsm8k_reward([prompt.messages], [[{"role": "assistant", "content": response}]],
                        answer=[prompt.reference])[0]
```

Spec file (3–10 real prompts per arm, with their reference answers):

```json
{"flag_rate": 0.05,
 "arms": {"gsm8k": {"verifier": "probe_adapters:gsm8k",
                    "prompts": [{"prompt_id": "1", "reference": "42",
                                 "messages": [{"role": "user", "content": "What is 6x7?"}]}]}}}
```

Run `curator-health preflight --spec spec.json` (exit 1 = some arm flagged). For every flagged probe,
show the user the junk response and the reward it got, then find the line in the verifier that lets
it through. That is the fix; the tool only finds it.

## Step 3 — in-run audit

**Live, TRL GRPOTrainer** (one-line change; reward values unchanged):

```python
from curator_health import RewardHealth
from curator_health.adapters.trl import RewardHealthCallback, wrap_trl_reward
health = RewardHealth(env_ids=[], group_size=G, mode="warn")   # arms self-register
trainer = GRPOTrainer(..., reward_funcs=wrap_trl_reward(reward_fn, health),
                      callbacks=[RewardHealthCallback(health)])
```
Flags appear in `log_history` as `health/flag/<env>` (and on stderr in `warn`/`gate`).
Wrap each reward function the user has; keep the user's G.

**Other trainers or finished runs (offline replay):** write one JSONL row per arm per round:

```json
{"env_id": "gsm8k", "n_prompts": 16, "n_rollouts": 128, "k_success": 40, "sum_score": 5.0, "sum_score_sq": 2.1}
```
`sum_score` / `sum_score_sq` are the sum and sum of squares of the **per-prompt pass rates** `k_i/G`
(success = reward ≥ 0.5). Then `curator-health audit --observations rounds.jsonl --group-size G`.

## Step 4 — report to the user

- Per arm: flagged or not, `D̂`, and degrees of freedom. The audit needs ≥ 40 df (≈ 40 prompts with
  mixed outcomes) before it can flag; say "not enough data yet" rather than "healthy" below that.
- Rounds where every rollout passes or every one fails carry no information and are skipped; an arm at
  0% or 100% pass rate cannot be audited — say so.
- A flagged arm: its reward does not depend on which question was asked. Recommend fixing or dropping
  it, and point out that advantage/learning-progress curricula (SEC, DUMP-style) actively **over-fund**
  such arms (measured 31–34% of questions on a random-reward arm vs 20% for a uniform mix).

## Blind spots — state them, never over-claim

- Does **not** catch a reward that depends on the prompt but rewards the wrong thing (wrong answer
  key, partial-credit bug). That needs held-out labelled checks or manual reading of ~50 scored samples.
- The probe only catches the junk patterns it tries.
- Assumes real arms differ in prompt difficulty; a dataset of near-identical prompts can look flat.

## On choosing the environment mix

Evidence from the CURATOR study (Qwen2.5-0.5B, 3 seeds): no mixing policy beat a uniform mix on
held-out accuracy (all 95% intervals contain 0). Default to uniform over the healthy arms; drop or fix
flagged arms. Do not promise accuracy gains from a scheduler.
