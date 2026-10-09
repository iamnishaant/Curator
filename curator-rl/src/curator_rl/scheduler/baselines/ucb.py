"""Standard UCB1 baseline (Roadmap Part J.1 #5).

Same proxy reward and the same softmax action map as Curator, but no
discounting (gamma = 1), no cost normalisation (cost_exponent = 0), no
status constraints and no calibration — it isolates the value of those components.
"""

from __future__ import annotations

from curator_rl.core.config import CalibCfg, ProxyCfg, SchedulerCfg, SignalsCfg
from curator_rl.scheduler.ducb import DiscountedUCB


class StandardUCB(DiscountedUCB):
    """UCB1 on the proxy reward: the discounting/cost/status ablation base."""

    def __init__(
        self,
        env_ids,
        signals_cfg: SignalsCfg,
        proxy_cfg: ProxyCfg,
        scheduler_cfg: SchedulerCfg,
        calib_cfg: CalibCfg,
        group_size: int,
        *,
        prompts_per_round: int | None = None,
    ) -> None:
        ablated = scheduler_cfg.model_copy(
            update={"gamma": 1.0, "cost_exponent": 0.0, "status_control": "off"}
        )
        super().__init__(
            env_ids, signals_cfg, proxy_cfg, ablated,
            calib_cfg.model_copy(update={"enabled": False}), group_size,
            prompts_per_round=prompts_per_round,
        )
