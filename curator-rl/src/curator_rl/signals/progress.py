"""Learning-progress estimators (Roadmap E.3, selection protocol on the simulator).

Three estimators share one observation feed (per-round posterior pass rate and
prompt count):

- LP-A: fast discounted rate minus slow discounted rate (conservative SE --
  the windows overlap, so se_f^2 + se_s^2 overstates the SE, never hides a
  real signal).
- LP-B: weighted-least-squares slope of the per-round posterior over the last
  W rounds, weights proportional to the prompt count; z = slope / SE. The
  residual variance inflation stands in for the cross-round autocorrelation
  of the pooled counts.
- LP-C: mean of the last W rounds minus the mean of the previous W rounds,
  independent-samples z-test. LP-C runs on the RAW per-round rate (k/n of
  that round alone), not the discounted pooled posterior: pooled windows
  overlap mechanically, which would invalidate the independent-samples
  assumption entirely (measured FPR 0.4 in validation vs <= 0.10 on raw).

Each round the engine calls `observe(env_id, posterior, n_prompts, raw_rate)`
(which records history only when the environment actually got prompts in
that round) and then `compute(env_id, posterior)` for the (lp, se, z) triple.
"""

from __future__ import annotations

import math

from curator_rl.signals.passrate import Posterior


class LPEstimator:
    """Selected-method LP estimator (lp_method: lp_a | lp_b | lp_c)."""

    def __init__(self, lp_method: str, window_rounds: int) -> None:
        if lp_method not in ("lp_a", "lp_b", "lp_c"):
            raise ValueError(f"unknown lp_method '{lp_method}'")
        if window_rounds < 1:
            raise ValueError("window_rounds must be >= 1")
        self._method = lp_method
        self._window = window_rounds
        # per-env history of (posterior pass rate, n_prompts), bounded at 2W
        # (LP-C needs the full window pair)
        self._history: dict[str, list[tuple[float, float, float]]] = {}

    # ------------------------------------------------------------------ update

    def observe(self, env_id: str, posterior: Posterior, n_prompts: int,
                raw_rate: float | None = None) -> None:
        if n_prompts <= 0:
            return
        if raw_rate is None:
            raw_rate = posterior.pass_rate
        history = self._history.setdefault(env_id, [])
        history.append((raw_rate, float(n_prompts), posterior.se_group))
        if len(history) > 2 * self._window:
            del history[: len(history) - 2 * self._window]

    def compute(self, env_id: str, posterior: Posterior) -> tuple[float, float, float]:
        """Return (lp, se, z) for one environment.

        A guard at insufficient history returns (0.0, 0.0, 0.0): no claim of
        progress is made while the estimator cannot decide anything.
        """
        if self._method == "lp_a":
            lp = posterior.rate_fast - posterior.rate_slow
            se = math.sqrt(posterior.se_fast**2 + posterior.se_slow**2)
            return lp, se, self._z(lp, se)

        history = self._history.get(env_id, [])
        if self._method == "lp_b":
            if len(history) < 3:
                return 0.0, 0.0, 0.0
            return self._slope_lp(history[-self._window :])
        # lp_c: non-overlapping windows, each of width W
        need = 2 * self._window
        if len(history) < need:
            return 0.0, 0.0, 0.0
        return self._windowed_means(history[-need:])

    # ------------------------------------------------------------- estimators

    @staticmethod
    def _z(lp: float, se: float) -> float:
        return lp / se if se > 0.0 else 0.0

    @staticmethod
    def _slope_lp(points: list[tuple[float, float, float]]) -> tuple[float, float, float]:
        """Two-stage WLS slope over (rate, n_prompts, se) triples, x = 0..k-1.

        Stage 1 fits with denominator weights to get the residual spread; stage 2
        re-weights each round by its combined variance (posterior se^2 inflated by
        the residual variance, which stands in for the cross-round autocorrelation
        of pooled discounted counts). The residual variance is divided by the
        degrees of freedom (k - 2), NOT by the weight total (which would shrink
        the SE by the prompt count and break the false-positive rate).
        """
        k = len(points)
        w = [max(p[1], 0.0) for p in points]
        total_w = sum(w)
        if total_w <= 0.0:
            return 0.0, 0.0, 0.0
        x = list(range(k))
        mean_x = sum(wi * xi for wi, xi in zip(w, x)) / total_w
        mean_y = sum(wi * p[0] for wi, p in zip(w, points)) / total_w
        sxx = sum(wi * (xi - mean_x) ** 2 for wi, xi in zip(w, x))
        if sxx <= 0.0:
            return 0.0, 0.0, 0.0

        def fit(weights: list[float]) -> tuple[float, float]:
            tw = sum(weights)
            mx = sum(wi * xi for wi, xi in zip(weights, x)) / tw
            my = sum(wi * p[0] for wi, p in zip(weights, points)) / tw
            sxx_w = sum(wi * (xi - mx) ** 2 for wi, xi in zip(weights, x))
            if sxx_w <= 0.0:
                return 0.0, mx
            slope_w = (
                sum(wi * (xi - mx) * (p[0] - my) for wi, xi, p in zip(weights, x, points)) / sxx_w
            )
            return slope_w, mx

        slope_stage1, _ = fit(w)
        resid_var = (
            sum(wi * (p[0] - (mean_y + slope_stage1 * (xi - mean_x))) ** 2
                for wi, xi, p in zip(w, x, points)) / max(k - 2, 1)
        )
        eff_var = [max(p[2], 1e-12) ** 2 + resid_var for p in points]  # se^2 + residual inflation

        slope, mx2 = fit([1.0 / v for v in eff_var])
        sxx_eff = sum((1.0 / v) * (xi - mx2) ** 2 for v, xi in zip(eff_var, x))
        se_slope = math.sqrt(1.0 / sxx_eff) if sxx_eff > 0.0 else 0.0
        z = slope / se_slope if se_slope > 0.0 else 0.0
        return slope, se_slope, z

    @staticmethod
    def _windowed_means(history: list[tuple[float, float, float]]) -> tuple[float, float, float]:
        w = len(history) // 2
        recent = [p[0] for p in history[w:]]
        previous = [p[0] for p in history[:w]]
        mean_r = sum(recent) / w
        mean_p = sum(previous) / w
        var_r = sum((v - mean_r) ** 2 for v in recent) / (w - 1)
        var_p = sum((v - mean_p) ** 2 for v in previous) / (w - 1)
        se = math.sqrt(max(var_r / w + var_p / w, 0.0))
        lp = mean_r - mean_p
        return lp, se, (lp / se if se > 0.0 else 0.0)

    # ------------------------------------------------------------- checkpoints

    def get_state(self) -> dict[str, object]:
        return {"method": self._method, "history": {e: list(h) for e, h in self._history.items()}}

    def load_checkpoint(self, state: dict[str, object]) -> None:
        self._history = {e: [tuple(p) for p in h] for e, h in state["history"].items()}  # type: ignore[misc,union-attr]
