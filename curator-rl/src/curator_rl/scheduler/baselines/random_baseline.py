"""Random-mixture baseline: Dirichlet(1) redrawn each round (Roadmap J.1 #3).

Module named `random_baseline` (not `random`) so it never shadows the
stdlib `random` module in imports.
"""

from __future__ import annotations

import numpy as np

from curator_rl.core.types import RoundObservation
from curator_rl.scheduler.base import BaseScheduler, Weights


class RandomScheduler(BaseScheduler):
    """Sanity floor: a fresh Dirichlet(1) mixture every round."""

    def __init__(self, env_ids, rng: np.random.Generator) -> None:
        super().__init__(env_ids)
        self._rng = rng

    def select_mixture(self, observation: RoundObservation | None) -> Weights:
        draws = self._rng.dirichlet(np.ones(len(self.env_ids)))
        return {env_id: float(w) for env_id, w in zip(self.env_ids, draws)}
