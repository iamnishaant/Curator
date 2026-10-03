"""Uniform baseline: w_i = 1/N (Roadmap Part J.1 #1, Phase 3 stub)."""

from __future__ import annotations

from curator_rl.core.types import RoundObservation
from curator_rl.scheduler.base import BaseScheduler, Weights


class UniformScheduler(BaseScheduler):
    """The default practice; also validates the static pipeline (Gate 3)."""

    def select_mixture(self, observation: RoundObservation | None) -> Weights:
        n = len(self.env_ids)
        return {env_id: 1.0 / n for env_id in self.env_ids}
