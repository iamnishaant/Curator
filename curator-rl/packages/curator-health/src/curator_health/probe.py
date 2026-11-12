"""Pre-flight reward probe: does an arm's reward pay for answers that cannot be right?

Verifier bugs become reward hacks under optimisation pressure: a checker that accepts the reference
number anywhere in the response, an empty box, or a program that exits before its tests run hands out
reward for junk, and RL learns to produce it ("Before the Model Learns the Bug: Fuzzing RLVR
Verifiers", 2026). Before any training, each arm's full reward path scores a fixed set of junk
responses on a few of its own prompts. Junk that earns reward is a finding, reported per probe.

This complements the in-run audit (``curator_health.audit``), which needs rounds of data and only sees
rewards that ignore the prompt; a buggy checker that still pays for correct answers passes the audit
but fails here. Costs seconds and no GPU. Pure stdlib.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

RewardFn = Callable[["ProbePrompt", str], float]


@dataclass(frozen=True)
class ProbePrompt:
    """A prompt as the probe needs it: the user text, the reference answer, and optional metadata."""

    prompt_id: str = ""
    messages: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    reference: str = ""
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def user_text(self) -> str:
        return next((str(m.get("content", "")) for m in self.messages if m.get("role") == "user"), "")


def _wrong_number(reference: str) -> str:
    try:
        value = float(str(reference).replace(",", ""))
    except ValueError:
        return "-999983"
    return str(int(value) + 1) if value.is_integer() else "-999983"


def probes(env_id: str, prompt: ProbePrompt) -> list[tuple[str, str]]:
    """(probe name, junk response) pairs for one prompt: none of them is a correct answer."""
    user = prompt.user_text
    common = [("empty", ""), ("echo_prompt", user)]
    if env_id == "mbpp":
        return common + [
            ("empty_code", "```python\n\n```"),
            ("pass_only", "```python\npass\n```"),
            ("exit_early", "```python\nimport sys\nsys.exit(0)\n```"),
            ("print_globals", "```python\nprint(globals())\n```"),
        ]
    if env_id == "countdown":
        out = common + [("empty_box", "\\boxed{}")]
        target, numbers = prompt.meta.get("target"), prompt.meta.get("numbers") or []
        if target is not None and target not in numbers:
            out.append(("target_literal", f"\\boxed{{{target}}}"))
        return out
    ref, wrong = str(prompt.reference), _wrong_number(prompt.reference)
    return common + [
        ("empty_box", "\\boxed{}"),
        ("wrong_box", f"\\boxed{{{wrong}}}"),
        ("number_spray", f"It is one of 0, 1, 2, {ref}, 10 or 100. \\boxed{{{wrong}}}"),
    ]


def probe_arm(env_id: str, reward_fn: RewardFn, prompts: Sequence[ProbePrompt], *,
              flag_rate: float = 0.05) -> dict[str, Any]:
    """Score the junk probes for one arm through its full reward path.

    A reward_fn that raises is counted as an error and treated as reward 0 for that probe: the probe
    must never propagate an exception into a training run.
    """
    per_probe: dict[str, list[float]] = {}
    errors = 0
    for prompt in prompts:
        for name, text in probes(env_id, prompt):
            try:
                reward = float(reward_fn(prompt, text))
            except Exception:  # noqa: BLE001 - fail-open: a probe problem must never stop a run
                errors += 1
                reward = 0.0
            per_probe.setdefault(name, []).append(reward)
    rates = {name: sum(v) / len(v) for name, v in per_probe.items()}
    n = sum(len(v) for v in per_probe.values())
    junk = sum(sum(v) for v in per_probe.values()) / n if n else 0.0
    flagged = bool(junk > flag_rate or any(v >= 0.5 for v in rates.values()))
    return {
        "junk_reward_rate": round(junk, 4),
        "probes": {k: round(v, 4) for k, v in rates.items()},
        "n_probes": n,
        "errors": errors,
        "flagged": flagged,
    }


def probe_all(specs: Mapping[str, tuple[RewardFn, Sequence[ProbePrompt]]], *,
              flag_rate: float = 0.05) -> dict[str, Any]:
    """Probe every arm. An arm that cannot be probed is reported with its error, never flagged."""
    report: dict[str, Any] = {"flag_rate": flag_rate, "arms": {}}
    for env_id, (reward_fn, prompts) in specs.items():
        try:
            report["arms"][env_id] = probe_arm(env_id, reward_fn, prompts, flag_rate=flag_rate)
        except Exception as exc:  # noqa: BLE001 - a probe problem must never stop a run
            report["arms"][env_id] = {"error": repr(exc)[:300], "flagged": False}
    report["flagged"] = sorted(e for e, r in report["arms"].items() if r.get("flagged"))
    return report
