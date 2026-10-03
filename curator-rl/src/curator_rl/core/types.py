"""Core data contracts (Roadmap B.3, frozen in Phase 2).

These types are owned by `core` (L0) so every layer (scheduler, simulator,
trainer adapter) can share them: the simulator, the trace replayer and the
real trainer all produce the same observation types.

Phase 2 introduces `Prompt`, `Verdict`, `RolloutGroup` and `CostPrior`.
Phase 3 adds `EnvRoundObs`, `RoundObservation` and `CalibrationObservation`
(Roadmap B.3): the simulator, the trace replayer and the real trainer all
produce these same observation types. `MixtureDecision` and `RoiRecord` are
added in the phases that first need them (Roadmap Part P).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SPLITS = ("train", "calib", "dev", "test")


@dataclass(frozen=True)
class Prompt:
    """A single training or evaluation item.

    `reference` is verifier-only: it MUST never be shown to the model and is
    not carried into the message list. `meta` holds environment-specific
    parameters (e.g. Countdown numbers and target) that the verifier needs.
    """

    prompt_id: str
    env_id: str
    split: str
    messages: tuple[dict[str, str], ...]
    reference: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join(m.get("content", "") for m in self.messages)


@dataclass(frozen=True)
class Verdict:
    """Outcome of verifying one completion against one prompt.

    Invariant: 0.0 <= score <= 1.0. `parse_ok` is False when no parseable
    answer could be extracted (a different failure than a wrong answer).
    """

    success: bool
    score: float
    parse_ok: bool
    verifier_seconds: float
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutGroup:
    """One prompt with its G rollouts. Produced by the reward wrapper /
    simulator, consumed by the Signal Engine through the round observation."""

    env_id: str
    prompt_id: str
    n: int
    prompt_tokens: int
    completion_tokens: tuple[int, ...]
    scores: tuple[float, ...]
    successes: tuple[bool, ...]
    verifier_seconds: float


@dataclass(frozen=True)
class CostPrior:
    """Cold-start cost estimate. NEVER used as a measured cost (Roadmap H)."""

    env_id: str
    per_prompt_usd: float
    note: str


@dataclass(frozen=True)
class EnvRoundObs:
    """Per-environment statistics for one round (Roadmap B.3).

    `sum_score`/`sum_score_sq` are over all rollout scores of the round;
    `n_groups_mixed` counts prompt groups with an informative pass rate
    (0 < k < G). `prompt_tokens`/`completion_tokens` are round totals.
    """

    env_id: str
    n_prompts: int
    n_rollouts: int
    k_success: int
    n_groups_mixed: int
    sum_score: float
    sum_score_sq: float
    prompt_tokens: int
    completion_tokens: int
    verifier_seconds: float
    gpu_seconds: float
    cost_usd: float


@dataclass(frozen=True)
class RoundObservation:
    """Everything the scheduler may see at a round boundary (Roadmap B.3/F.1).

    The scheduler MUST NOT be able to read any hidden simulator variable, so
    this type carries only measured quantities. `budget_remaining_usd` is
    tracked by the caller (harness or trainer callback) under the ledger rule
    (Roadmap H.6).
    """

    round: int
    steps: int
    per_env: dict[str, EnvRoundObs]
    weights_used: dict[str, float]
    round_cost_usd: float
    overhead_usd: float
    budget_remaining_usd: float


@dataclass(frozen=True)
class CalibrationObservation:
    """One calibration-benchmark evaluation of a window (Roadmap B.3, I.1/I.3).

    `exposure_by_env` is the compute mass W_{j,k} (prompt-equivalents) each
    environment received since the previous calibration; `window_cost_usd`
    is the charged training cost of that window.
    """

    window_k: int
    round: int
    score_total: float
    score_by_domain: dict[str, float]
    se_total: float
    se_by_domain: dict[str, float]
    n_items: int
    eval_cost_usd: float
    exposure_by_env: dict[str, float]
    window_cost_usd: float
