"""Curator facade (Roadmap Part P v1 Phase 5 class list, A.2 decision 7).

Full Curator = D-UCB + cost normalisation + status constraints, with the
calibration hooks in place but the alpha/beta fit deferred to Phase 9 (the
Phase 5 version is "Curator v0": proxy + cost + discounted UCB).

Extra surface beyond `BaseScheduler`:
- `compute_scores()` — the current mu_hat + bonus per env (debug/ablations).
- `get_roi()` — stub returning None until the ROI engine (Phase 10).
- `last_decision` — the round's `MixtureDecision` (B.3; the D-26 deferral ends).
"""

from __future__ import annotations

from collections.abc import Mapping

from curator_rl.core.types import CalibrationObservation, RoundObservation
from curator_rl.scheduler.base import Weights
from curator_rl.scheduler.ducb import DiscountedUCB


class Curator(DiscountedUCB):
    """Curator v0/full facade over the D-UCB backend (Roadmap Part E/J)."""

    def compute_scores(self) -> dict[str, float]:
        """Current UCB score components (mu_hat + bonus) per environment."""
        return super().compute_scores()

    def select_mixture(self, observation: RoundObservation | None) -> Weights:
        return super().select_mixture(observation)

    def update_observation(self, observation: RoundObservation) -> None:
        super().update_observation(observation)

    def update_calibration(self, observation: CalibrationObservation) -> None:
        # stores the observation on the engine; the alpha/beta fit + S5
        # mismatch feed arrive with Phase 9 (D-34)
        super().update_calibration(observation)

    def get_roi(self) -> Mapping[str, float] | None:
        """ROI leaderboard stub — the ROI engine arrives in Phase 10."""
        return None
