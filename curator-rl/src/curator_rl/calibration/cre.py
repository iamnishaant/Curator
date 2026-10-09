"""Calibrated Reward Engine (CRE): one Gaussian posterior per environment (D-76).

Latent quantity: b_j = expected change of environment j's OWN calibration slice
per training dollar charged to j, under the mixtures actually run. It is an
observational estimate; transfer from other arms and re-evaluation churn are
inside it, and it is not a causal effect (D-76, attribution caveat).

Evidence (likelihood): each time j's slice is re-evaluated, the paired change
since its last evaluation is one observation `gain = b_j * usd + noise(se)`,
where `usd` is the dollars the ledger charged to j over that span and `se` the
paired (McNemar) standard error. Cost is a MEASURED regressor here, never a
noisy denominator of the evidence.

Prior: `b_j ~ N(m0_j, v0_j)` with `m0_j = s * x_j / c_j` (x_j: current proxy
signal alpha*LP' + beta*SR, c_j: measured unit cost) and `s = cre.proxy_scale`,
a constant FROZEN from tuning seeds (D-78): re-estimating it online from the one
or two arms that targeted calibration evaluates was fragile (a junk arm dragged
it to ~0). The online median `median_j(b_hat_j * c_j / xbar_j)` is kept as a
diagnostic (`online_scale`) only. The prior variance is
`(rho * max(|m0_j|, floor))^2` with rho frozen from tuning seeds.
Arms without evidence therefore sit exactly at their proxy prior; there is no
separate fallback path once the engine is active.

Reward: `r_j = clip(m_j * B / R_max, 0, 1)`, a FIXED per-arm transform (no
per-round scaling, no dependence on other arms; B = run budget, R_max frozen
from tuning seeds). The posterior SD is exposed for logging and is NOT used as
an exploration bonus (no double counting with the D-UCB count bonus).

Modes: `full` (precision-weighted posterior) and `no_uncertainty` (ablation:
the likelihood-only point estimate when evidence exists, else the prior mean).
`discount < 1` forgets old evidence per calibration window (ablation).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass

from curator_rl.core.config import CreCfg

_EPS = 1e-12
_FLOOR_FRAC = 0.05  # floor of the prior scale, as a fraction of R_max / B


@dataclass(frozen=True)
class CreEstimate:
    """One environment's current estimate (logged with every reward)."""

    env_id: str
    reward: float
    mean_per_usd: float
    sd_per_usd: float
    prior_mean: float
    n_obs: int
    source: str  # "prior" (no evidence for this arm) or "posterior"


class CalibratedRewardEngine:
    """Pure, deterministic, checkpointable (no RNG)."""

    def __init__(self, env_ids, cfg: CreCfg) -> None:
        self.env_ids = tuple(sorted(env_ids))
        self._cfg = cfg
        self._prec = {e: 0.0 for e in self.env_ids}   # sum usd^2 / se^2 (discounted)
        self._num = {e: 0.0 for e in self.env_ids}    # sum usd * gain / se^2 (discounted)
        self._n = {e: 0 for e in self.env_ids}
        self.scale = float(cfg.proxy_scale)           # frozen constant (D-78)
        self.online_scale = 0.0                       # diagnostic only, never used for rewards
        self.active = False
        self.log: list[dict[str, object]] = []

    # ------------------------------------------------------------------ evidence

    def b_hat(self, env_id: str) -> float | None:
        return self._num[env_id] / self._prec[env_id] if self._prec[env_id] > 0 else None

    def ingest(
        self,
        span_gain: Mapping[str, Mapping[str, float]],
        proxy_x: Mapping[str, float],
        unit_cost: Mapping[str, float],
        *,
        ready: bool,
        window_k: int = 0,
    ) -> None:
        """Book one calibration window: span gains per evaluated arm (D-76)."""
        lam = self._cfg.discount
        if lam < 1.0:
            for e in self.env_ids:
                self._prec[e] *= lam
                self._num[e] *= lam
        for e in self.env_ids:
            g = span_gain.get(e)
            if g is None or g["usd"] <= _EPS:
                continue
            se2 = max(float(g["se"]), 1e-6) ** 2
            self._prec[e] += float(g["usd"]) ** 2 / se2
            self._num[e] += float(g["usd"]) * float(g["gain"]) / se2
            self._n[e] += 1
        ratios = []
        for e in self.env_ids:
            b = self.b_hat(e)
            x, c = float(proxy_x.get(e, 0.0)), float(unit_cost.get(e, 0.0))
            if b is not None and x > 1e-9 and c > 0.0:
                ratios.append(b * c / x)
        if ratios:
            self.online_scale = max(statistics.median(ratios), 0.0)
        # active once the first READY calibration has evidence for at least one arm
        if ready and any(self._n[e] > 0 for e in self.env_ids):
            self.active = True
        self.log.append({
            "event": "ingest", "window_k": window_k, "scale": self.scale,
            "online_scale": self.online_scale, "active": self.active,
            "evidence": {e: {"n": self._n[e], "b_hat": self.b_hat(e)} for e in self.env_ids},
        })

    # ------------------------------------------------------------------- reward

    def estimate(self, env_id: str, x_now: float, c_now: float, budget_usd: float) -> CreEstimate:
        cfg = self._cfg
        b_total = max(budget_usd, _EPS)
        floor = _FLOOR_FRAC * cfg.roi_scale / b_total
        m0 = self.scale * x_now / c_now if c_now > _EPS else 0.0
        v0 = (cfg.prior_rel_sd * max(abs(m0), floor)) ** 2
        prec_d, num_d = self._prec[env_id], self._num[env_id]
        if cfg.mode == "no_uncertainty":
            mean = num_d / prec_d if prec_d > 0 else m0
            sd = 1.0 / math.sqrt(prec_d) if prec_d > 0 else math.sqrt(v0)
        else:
            precision = 1.0 / v0 + prec_d
            mean = (m0 / v0 + num_d) / precision
            sd = 1.0 / math.sqrt(precision)
        reward = min(max(mean * b_total / cfg.roi_scale, 0.0), 1.0)
        return CreEstimate(
            env_id=env_id, reward=reward, mean_per_usd=mean, sd_per_usd=sd, prior_mean=m0,
            n_obs=self._n[env_id], source="posterior" if prec_d > 0 else "prior",
        )

    # --------------------------------------------------------------- checkpoints

    def get_state(self) -> dict[str, object]:
        return {"prec": dict(self._prec), "num": dict(self._num), "n": dict(self._n),
                "online_scale": self.online_scale, "active": self.active}

    def load_checkpoint(self, state: Mapping[str, object]) -> None:
        self._prec = {e: float(v) for e, v in state["prec"].items()}  # type: ignore[union-attr]
        self._num = {e: float(v) for e, v in state["num"].items()}  # type: ignore[union-attr]
        self._n = {e: int(v) for e, v in state["n"].items()}  # type: ignore[union-attr]
        self.online_scale = float(state.get("online_scale", 0.0))  # type: ignore[arg-type]
        self.active = bool(state["active"])
