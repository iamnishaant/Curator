"""DUMP-style bandit baseline (Roadmap v3 section 4.4; Wang et al., DUMP).

UCB over distributions (environments) with the mean absolute GRPO advantage as
the arm reward: `score_i = mean_i + c * sqrt(ln(max(N, e)) / n_i)` with
undiscounted prompt counts, then `softmax(score / tau)` with the shared
exploration floor. A re-implementation from the paper's description ("style"):
the exact windowing and constants of the original are not reproduced.
Cost-blind and uncalibrated by design.
"""

from __future__ import annotations

import math

from curator_rl.core.config import DumpBaselineCfg, SchedulerCfg
from curator_rl.core.types import RoundObservation
from curator_rl.scheduler.base import BaseScheduler, validate_weights
from curator_rl.scheduler.mixture import softmax_floor


class DUMPStyleUCB(BaseScheduler):
    """UCB1 on mean |advantage| per prompt, softmax action map."""

    def __init__(self, env_ids, dump_cfg: DumpBaselineCfg, scheduler_cfg: SchedulerCfg) -> None:
        super().__init__(env_ids)
        self._cfg = dump_cfg
        self._sched = scheduler_cfg
        self._n = {e: 0.0 for e in self.env_ids}      # prompts seen
        self._sum = {e: 0.0 for e in self.env_ids}    # sum of per-prompt mean |A|
        self._round = 0

    def scores(self) -> dict[str, float]:
        total = sum(self._n.values())
        log_arg = math.log(max(total, math.e))
        out: dict[str, float] = {}
        for e in self.env_ids:
            n = max(self._n[e], 1e-9)
            out[e] = self._sum[e] / n + self._cfg.exploration_coef * math.sqrt(log_arg / n)
        return out

    def select_mixture(self, observation: RoundObservation | None) -> dict[str, float]:
        self._round += 1
        if self._round <= self._sched.warmup_rounds or observation is None:
            return {e: 1.0 / len(self.env_ids) for e in self.env_ids}
        weights = softmax_floor(
            self.scores(), self._cfg.tau, self._sched.epsilon, list(self.env_ids), score_norm="none"
        )
        validate_weights(weights, self.env_ids)
        return weights

    def update_observation(self, observation: RoundObservation) -> None:
        for env_id, obs in observation.per_env.items():
            if obs.n_prompts > 0:
                self._n[env_id] += obs.n_prompts
                self._sum[env_id] += obs.sum_abs_adv

    def get_state(self) -> dict[str, object]:
        return {"round": self._round, "n": dict(self._n), "sum": dict(self._sum)}

    def load_checkpoint(self, state) -> None:
        self._round = int(state["round"])  # type: ignore[arg-type]
        self._n = {e: float(v) for e, v in state["n"].items()}  # type: ignore[union-attr]
        self._sum = {e: float(v) for e, v in state["sum"].items()}  # type: ignore[union-attr]
