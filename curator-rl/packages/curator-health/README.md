# curator-health

**A drop-in, label-free guardrail for RLVR: does your reward actually depend on the prompt?**

Reinforcement learning with verifiable rewards (RLVR) uses a program as the reward — a math checker, a
unit test, a schema validator. That makes the reward partly a *software artifact*, and if it is wrong,
optimisation learns the bug. `curator-health` answers the question **"does this reward mean
anything?"**, before and during training, with **no labels, no gradients, no reference verifier and no
extra GPU**.

It never replaces your reward function and never changes the value your trainer sees. It observes.

## The two checks

| Check | When | How |
|---|---|---|
| **Pre-flight probe** | before training | score a fixed list of junk answers (empty, echoed prompt, empty/wrong `\boxed{}`, number spray, code that exits early or prints globals) through each arm's full reward path. Junk that earns reward is a finding. |
| **In-run audit** | during training, free | a verifier's reward depends on the prompt, and prompts differ in difficulty, so group pass rates are *over-dispersed*: `Var(k/G) = p(1−p)/G · D` with `D = (G+κ)/(κ+1)`. A reward that **ignores** the prompt has `D ≈ 1`. Flag when the upper confidence bound of `log D̂` falls below `log d_max`. |

Measured (Qwen2.5 0.5B/1.5B, 64 prompts × G = 8): real verifiers **D = 3.0–4.4**; a random reward
**D = 1.0–1.09**. The audit flagged the random-reward arm in **6/6** GPU runs and **never** a real arm.

## Quickstart

```bash
pip install curator-health      # stdlib only; nothing added to a training environment
```

```python
from curator_health import RewardHealth, HealthConfig, ProbePrompt

health = RewardHealth(env_ids=["gsm8k", "mbpp", "noisy"], group_size=8,
                      config=HealthConfig(d_max=1.5), mode="shadow")

# 1) once, before training: probe each arm's reward path with junk answers
health.preflight({
    "gsm8k": (gsm8k_reward, [ProbePrompt(prompt_id="1", reference="42",
                                         messages=({"role": "user", "content": "What is 6x7?"},))]),
})

# 2) every round, per arm: feed the moments you already have
health.observe(env_id="gsm8k", n_prompts=16, n_rollouts=128, k_success=40,
               sum_score=5.0, sum_score_sq=2.1)

print(health.flagged())     # ['noisy']
print(health.report())      # JSON-serialisable snapshot for your logs
```

### Modes

| Mode | Behaviour |
|---|---|
| `shadow` (default) | observe, flag, report — offers the host no way to act |
| `warn` | as shadow, but surface `flags()`/`report()` loudly |
| `gate` | as warn; the host reads `flags()` and acts (drop an arm, alert, …) |

### CLI

```bash
curator-health preflight --spec spec.json        # probe verifiers (exit 1 if any arm is flagged)
curator-health audit --observations rounds.jsonl # replay round moments through the in-run audit
```

## Why it is safe to adopt

- **Fail-open.** Every entry point counts and swallows its own errors; a malformed observation or a
  raising verifier can never reach your training run. The reward path is never modified.
- **Zero GPU cost.** The in-run audit is arithmetic on `n_prompts`, `n_rollouts`, `k_success`,
  `sum_score`, `sum_score_sq` — moments any GRPO trainer already has.
- **Deterministic and checkpointable** — identical flags after a resume.

## Blind spots (read this)

- The in-run audit catches rewards that **ignore the prompt**. It does **not** catch a reward that
  depends on the prompt but rewards the wrong thing (a wrong answer key) — verifier-only feedback
  provably cannot resolve that (arXiv 2609.35677); that is held-out calibration's job.
- The probe only catches the junk patterns it tries.
- The audit assumes real arms differ in prompt difficulty (κ ≲ 5).

## Relationship to similar tools

- **VerifierFuzz** asks *"does my verifier disagree with an independent reference verifier?"*
  — offline, differential, needs a reference. Ours asks *"does my reward depend on the question at
  all?"* — online, needs no reference. **Complementary, not competing.**
- Verifier fuzzing papers check **before** training only; gradient-based detectors need gradients and
  labels. This needs neither.

## What it is not

Not a verifier, not a trainer, not a scheduler, and not a fix for every reward-hacking mode. It is a
tripwire.

MIT licensed.
