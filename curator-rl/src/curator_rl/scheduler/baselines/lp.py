"""Learning-progress curriculum baseline (Roadmap Part J.1 #4, Graves/TSCL style).

Boltzmann over |LP| from the same SignalEngine (its own engine instance runs
`lp_use: abs`), cost-blind, with the same exploration floor. Isolates the
value of cost awareness: identical signals, no cost normalisation, no
calibration, no status constraints.
"""

from __future__ import annotations

from curator_rl.core.config import CalibCfg, ProxyCfg, SchedulerCfg, SignalsCfg
from curator_rl.core.types import CalibrationObservation, RoundObservation
from curator_rl.scheduler.base import BaseScheduler, validate_weights
from curator_rl.scheduler.mixture import scores_to_weights
from curator_rl.signals.engine import SignalEngine


class LPCurriculum(BaseScheduler):
    """Softmax over the absolute learning progress, cost-blind (J.1 #4)."""

    def __init__(
        self,
        env_ids,
        signals_cfg: SignalsCfg,
        proxy_cfg: ProxyCfg,
        scheduler_cfg: SchedulerCfg,
        calib_cfg: CalibCfg,
        group_size: int,
    ) -> None:
        super().__init__(env_ids)
        abs_signals = signals_cfg.model_copy(update={"lp_use": "abs"})
        self._engine = SignalEngine(
            abs_signals, proxy_cfg, calib_cfg,
            scheduler_cfg.cost_exponent, group_size, env_ids,
        )
        self.cfg = scheduler_cfg
        self._latest: dict[str, float] = {}
        self._round = 0

    def select_mixture(self, observation: RoundObservation | None) -> dict[str, float]:
        self._round += 1
        if self._round <= self.cfg.warmup_rounds or observation is None or not self._latest:
            return {e: 1.0 / len(self.env_ids) for e in self.env_ids}
        scores = {e: abs(self._latest[e]) for e in self.env_ids}
        weights = scores_to_weights(
            scores, self.cfg.tau, self.cfg.epsilon, list(self.env_ids),
            score_norm=self.cfg.score_norm,
        )
        validate_weights(weights, self.env_ids)
        return weights

    def update_observation(self, observation: RoundObservation) -> None:
        vectors = self._engine.update(observation)
        self._latest = {e: float(sv.lp) for e, sv in vectors.items()}

    def update_calibration(self, observation: CalibrationObservation) -> None:
        self._engine.update_calibration(observation)

    def get_state(self) -> dict[str, object]:
        return {"round": self._round, "latest": dict(self._latest), "engine": self._engine.get_state()}

    def load_checkpoint(self, state) -> None:
        self._round = int(state["round"])  # type: ignore[arg-type]
        self._latest = {e: float(v) for e, v in state["latest"].items()}  # type: ignore[union-attr]
        self._engine.load_checkpoint(state["engine"])  # type: ignore[arg-type]
