"""Window gains and credit rules (Roadmap Part I.2-I.3, D-73).

Pure functions and small accumulators; no RNG, no scheduler state.

- `window_deltas`: per-slice change of the calibration benchmark between two
  evaluations, with its standard error (the paired SE when the observation
  carries one, else the independent-samples SE).
- `share_credit` (C1): the proposal's credit, delta_S x compute share. Kept as the
  baseline rule and for logging; it carries no information about WHICH
  environment caused the gain (I.2).
- `OwnSliceCredit` (C2, diagonal form): for environment j with its own benchmark
  slice d(j), the model is delta_S_{d(j),k} = W_{j,k} * b_j + noise(SE_k), with
  W_{j,k} the share of window k's compute spent on j. Weighted least squares
  across windows with an optional Gaussian prior on b_j. Transfer from other
  environments into j's slice is not modelled here (C2d is the full matrix form).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from curator_rl.core.types import CalibrationObservation

_MIN_SE = 1e-6


def window_deltas(
    prev: CalibrationObservation, cur: CalibrationObservation
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-domain change cur - prev and its SE (Roadmap I.3)."""
    delta: dict[str, float] = {}
    se: dict[str, float] = {}
    paired = cur.delta_se_by_domain or {}
    for d, s in cur.score_by_domain.items():
        if d not in prev.score_by_domain:
            continue
        delta[d] = float(s) - float(prev.score_by_domain[d])
        if d in paired:
            se[d] = max(float(paired[d]), _MIN_SE)
        else:
            se[d] = max(math.hypot(float(cur.se_by_domain.get(d, 0.0)),
                                   float(prev.se_by_domain.get(d, 0.0))), _MIN_SE)
    return delta, se


def exposure_shares(exposure: Mapping[str, float]) -> dict[str, float]:
    """W_{j,k}: share of the window's compute mass spent on each environment."""
    total = sum(max(float(v), 0.0) for v in exposure.values())
    if total <= 0:
        return {e: 0.0 for e in exposure}
    return {e: max(float(v), 0.0) / total for e, v in exposure.items()}


def share_credit(delta_total: float, exposure: Mapping[str, float]) -> dict[str, float]:
    """C1: credit_j = delta_S * share_j (Roadmap I.2; baseline rule only)."""
    return {e: delta_total * w for e, w in exposure_shares(exposure).items()}


class OwnSliceCredit:
    """Weighted least squares of own-slice gains on compute share, per environment."""

    def __init__(self, env_ids) -> None:
        self.env_ids = tuple(env_ids)
        self._sww = {e: 0.0 for e in self.env_ids}   # sum W^2 / SE^2
        self._swy = {e: 0.0 for e in self.env_ids}   # sum W * y / SE^2
        self._n = {e: 0 for e in self.env_ids}       # windows with exposure

    def add(self, env_id: str, share: float, gain: float, se: float) -> None:
        if share <= 0.0:
            return
        inv = 1.0 / max(se, _MIN_SE) ** 2
        self._sww[env_id] += share * share * inv
        self._swy[env_id] += share * gain * inv
        self._n[env_id] += 1

    def estimate(self, env_id: str, *, prior_mean: float = 0.0,
                 prior_sd: float | None = None) -> tuple[float, float]:
        """Posterior (mean, sd) of b_j; `prior_sd=None` is the likelihood-only estimate."""
        precision = self._sww[env_id]
        weighted = self._swy[env_id]
        if prior_sd is not None and prior_sd > 0:
            precision += 1.0 / prior_sd**2
            weighted += prior_mean / prior_sd**2
        if precision <= 0.0:
            return prior_mean, float("inf")
        return weighted / precision, 1.0 / math.sqrt(precision)

    def windows(self, env_id: str) -> int:
        return self._n[env_id]

    def get_state(self) -> dict[str, object]:
        return {"sww": dict(self._sww), "swy": dict(self._swy), "n": dict(self._n)}

    def load_checkpoint(self, state: Mapping[str, object]) -> None:
        self._sww = {e: float(v) for e, v in state["sww"].items()}  # type: ignore[union-attr]
        self._swy = {e: float(v) for e, v in state["swy"].items()}  # type: ignore[union-attr]
        self._n = {e: int(v) for e, v in state["n"].items()}  # type: ignore[union-attr]
