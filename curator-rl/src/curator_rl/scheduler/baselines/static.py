"""Static-mixture baseline (Roadmap Part J.1 #2).

Fixed weights set before training and recorded (size-proportional is the
special case of equal weights). In the simulator the weights come from the
benchmark slice weights pi_d (D-45) — visible before training, never a
hidden training signal.
"""

from __future__ import annotations

from collections.abc import Mapping

from curator_rl.core.types import RoundObservation
from curator_rl.scheduler.base import BaseScheduler, validate_weights


class StaticMixtureScheduler(BaseScheduler):
    """Returns the same pre-registered mixture every round."""

    def __init__(self, env_ids, weights: Mapping[str, float]) -> None:
        super().__init__(env_ids)
        validate_weights(dict(weights), self.env_ids)
        self._weights = {e: float(weights[e]) for e in self.env_ids}

    def select_mixture(self, observation: RoundObservation | None) -> dict[str, float]:
        return dict(self._weights)

    def get_state(self) -> dict[str, object]:
        return {"weights": dict(self._weights)}

    def load_checkpoint(self, state) -> None:
        validate_weights(dict(state["weights"]), self.env_ids)  # type: ignore[arg-type]
        self._weights = {e: float(v) for e, v in state["weights"].items()}  # type: ignore[union-attr]
