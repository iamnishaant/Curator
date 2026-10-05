"""Signal-richness estimation (Roadmap E.3 item 3, S-8 resolution).

SR is defined at group level. Three modes:

- `mixed`: fraction of prompt groups with 0 < k < G. Exact from the round
  aggregate (`n_groups_mixed` / `n_prompts`).
- `band`: fraction of groups with band_lo <= k/G <= band_hi. The round
  aggregate only carries the 0<k<G count, so for band (and variance) the
  per-prompt band membership is estimated from the posterior interval
  [p_lo, p_hi] overlap with the band and smoothed. For the MVP settings
  (G = 8 <= 9) the roadmap's 0.1-0.9 band collapses to `mixed` anyway (S-8).
- `variance`: group reward std above `variance_min`, decided on the
  discounted per-round group-rate variance and emitted as 0/1.

All modes discount their per-round raw value with the signal lambda before
it is stored (D-31), and every round without prompts leaves the stored value
untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from curator_rl.core.config import RichnessCfg
from curator_rl.core.types import EnvRoundObs
from curator_rl.signals.passrate import Posterior


@dataclass
class _EnvState:
    ew_value: float = 0.0
    ew_sq: float = 0.0  # group-rate second moment, variance mode
    ew_var: float = 0.0  # group-rate variance, variance mode (Welford-free)


@dataclass
class _Store:
    lam: float
    mode: str
    band_lo: float
    band_hi: float
    variance_min: float
    envs: dict[str, _EnvState] = field(default_factory=dict)


class RichnessEstimator:
    """Per-environment signal-richness estimator (Roadmap E.3.3)."""

    def __init__(self, cfg: RichnessCfg, lam: float, group_size: int) -> None:
        self._store = _Store(
            lam=lam, mode=str(cfg.mode), band_lo=cfg.band_lo, band_hi=cfg.band_hi,
            variance_min=cfg.variance_min,
        )
        self.group_size = group_size
        self.cfg = cfg

    # ------------------------------------------------------------------ update

    def update(self, env_id: str, obs: EnvRoundObs, posterior: Posterior) -> float:
        """Fold one round in; returns the stored richness value (or 0 when unset)."""
        state = self._store.envs.setdefault(env_id, _EnvState())
        if obs.n_prompts <= 0:
            return state.ew_value
        lam = self._store.lam
        if self._store.mode == "mixed":
            raw = obs.n_groups_mixed / obs.n_prompts
        elif self._store.mode == "band":
            raw = self._band_overlap(posterior.pass_lo, posterior.pass_hi)
        else:  # variance: group-rate std from the round's score moments
            mean_rate = obs.sum_score / obs.n_prompts
            var_rate = max(obs.sum_score_sq / obs.n_prompts - mean_rate**2, 0.0)
            state.ew_var = lam * state.ew_var + (1.0 - lam) * var_rate
            raw = 1.0 if math.sqrt(state.ew_var) > self._store.variance_min else 0.0
        state.ew_value = lam * state.ew_value + (1.0 - lam) * raw
        return state.ew_value

    def update_groups(self, env_id: str, rates: list[float]) -> float:
        """Exact per-group path (unit tests; available where raw groups exist)."""
        if not rates:
            return self._store.envs.get(env_id, _EnvState()).ew_value
        state = self._store.envs.setdefault(env_id, _EnvState())
        if self._store.mode == "mixed":
            raw = sum(1.0 for r in rates if 0.0 < r < 1.0) / len(rates)
        elif self._store.mode == "band":
            raw = sum(1.0 for r in rates if self._store.band_lo <= r <= self._store.band_hi) / len(rates)
        else:
            mean = sum(rates) / len(rates)
            var = sum((r - mean) ** 2 for r in rates) / len(rates)
            state.ew_var = self._store.lam * state.ew_var + (1.0 - self._store.lam) * var
            raw = 1.0 if math.sqrt(state.ew_var) > self._store.variance_min else 0.0
        state.ew_value = self._store.lam * state.ew_value + (1.0 - self._store.lam) * raw
        return state.ew_value

    def value(self, env_id: str) -> float:
        return self._store.envs.get(env_id, _EnvState()).ew_value

    def _band_overlap(self, lo: float, hi: float) -> float:
        """Fraction of [p_lo, p_hi] covered by [band_lo, band_hi]."""
        if hi <= lo or self._store.band_hi <= self._store.band_lo:
            return 0.0
        overlap = min(hi, self._store.band_hi) - max(lo, self._store.band_lo)
        return max(overlap, 0.0) / (hi - lo)

    # ------------------------------------------------------------- checkpoints

    def get_state(self) -> dict[str, list[float]]:
        return {
            env: [st.ew_value, st.ew_sq, st.ew_var] for env, st in self._store.envs.items()
        }

    def load_checkpoint(self, state: dict[str, list[float]]) -> None:
        self._store.envs = {e: _EnvState(ew_value=v[0], ew_sq=v[1], ew_var=v[2]) for e, v in state.items()}
