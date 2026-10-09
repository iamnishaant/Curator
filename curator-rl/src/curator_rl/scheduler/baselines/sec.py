"""SEC-style bandit baseline (Roadmap v3 section 4.4, J.1 #9; Chen et al., Self-Evolving Curriculum).

Arms are environments. The arm reward is the environment's mean absolute GRPO
advantage over the round (`EnvRoundObs.sum_abs_adv / n_prompts`), a proxy for
immediate learning gain. The value estimate is updated with the TD(0) rule
`Q <- alpha * r + (1 - alpha) * Q` for arms that received prompts, and the
mixture is `softmax(Q / tau)` with the shared exploration floor.

This is a re-implementation from the paper's description ("style"), not a
reproduction of the authors' code. It is cost-blind and uncalibrated by design:
it isolates what the learnability signal alone buys. Same warm-up and floor as
every other adaptive method (protocol constants, D-68).
"""

from __future__ import annotations

from curator_rl.core.config import SchedulerCfg, SecBaselineCfg
from curator_rl.core.types import RoundObservation
from curator_rl.scheduler.base import BaseScheduler, validate_weights
from curator_rl.scheduler.mixture import softmax_floor


class SECStyleBandit(BaseScheduler):
    """TD(0) value per environment on mean |advantage|; Boltzmann sampling."""

    def __init__(self, env_ids, sec_cfg: SecBaselineCfg, scheduler_cfg: SchedulerCfg) -> None:
        super().__init__(env_ids)
        self._cfg = sec_cfg
        self._sched = scheduler_cfg
        self._q = {e: 0.0 for e in self.env_ids}
        self._round = 0

    def select_mixture(self, observation: RoundObservation | None) -> dict[str, float]:
        self._round += 1
        if self._round <= self._sched.warmup_rounds or observation is None:
            return {e: 1.0 / len(self.env_ids) for e in self.env_ids}
        weights = softmax_floor(
            self._q, self._cfg.tau, self._sched.epsilon, list(self.env_ids), score_norm="none"
        )
        validate_weights(weights, self.env_ids)
        return weights

    def update_observation(self, observation: RoundObservation) -> None:
        alpha = self._cfg.alpha
        for env_id, obs in observation.per_env.items():
            if obs.n_prompts <= 0:
                continue  # bandit feedback: only pulled arms are updated
            reward = obs.sum_abs_adv / obs.n_prompts
            self._q[env_id] = alpha * reward + (1.0 - alpha) * self._q[env_id]

    def get_state(self) -> dict[str, object]:
        return {"round": self._round, "q": dict(self._q)}

    def load_checkpoint(self, state) -> None:
        self._round = int(state["round"])  # type: ignore[arg-type]
        self._q = {e: float(v) for e, v in state["q"].items()}  # type: ignore[union-attr]
