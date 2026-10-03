"""Episode harness and oracle-gap metric (Roadmap F.5, v1 Phase 3).

`run_episode` drives one scheduler through one scenario world under the
budget-ledger stop rule (Roadmap H.6/S-13, at round granularity): the round
that crosses the budget is charged and the episode stops right after it, so
the overshoot is at most one round. Reporting evaluations (final benchmark)
are uncharged; scheduler-owned calibration evaluations (optional) are charged.

Seeds: `SeedManager` hands out independent named streams, so the world,
the scheduler and the oracle see independent randomness for the same seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Protocol

from curator_rl.core.types import CalibrationObservation, RoundObservation
from curator_rl.scheduler.base import BaseScheduler
from curator_rl.simulator.scenarios import ScenarioCfg, build_world
from curator_rl.simulator.world import SimWorld


class OracleScheduler(Protocol):
    """Duck type shared by oracles: they read hidden world state by design."""

    def select_mixture(self, observation: RoundObservation | None) -> dict[str, float]: ...


SchedulerLike = BaseScheduler | OracleScheduler


@dataclass
class EpisodeResult:
    """Outcome of one simulated episode (Roadmap F.5 outputs)."""

    scenario_id: str
    method: str
    seed: int
    rounds: int
    final_score: float
    final_score_se: float
    total_cost_usd: float
    budget_usd: float
    benchmark_evals: int
    round_logs: list[dict] = field(default_factory=list)
    calib_logs: list[dict] = field(default_factory=list)

    @property
    def within_budget_tolerance(self) -> bool:
        """Overshoot is at most one round (Roadmap H.6)."""
        return self.total_cost_usd <= self.budget_usd * 1.5


def apply_scenario_constraints(
    weights: dict[str, float], world: SimWorld, round_t: int
) -> dict[str, float]:
    """Zero out not-yet-available (late-start) envs and renormalise (S-H)."""
    masked = {
        env_id: (0.0 if env.late_start_round > round_t else weights.get(env_id, 0.0))
        for env_id, env in world.envs.items()
    }
    total = sum(masked.values())
    if total <= 0:
        avail = [i for i, e in world.envs.items() if e.late_start_round <= round_t]
        n = max(len(avail), 1)
        return {i: (1.0 / n if i in avail else 0.0) for i in masked}
    return {i: v / total for i, v in masked.items()}


def run_episode(
    scheduler: SchedulerLike,
    scenario: ScenarioCfg,
    seed: int,
    *,
    method: str = "unknown",
) -> EpisodeResult:
    """Run one scheduler on a fresh (scenario, seed) world until the budget
    is exhausted or `max_rounds` is reached (Roadmap F.5)."""
    world = build_world(scenario, seed)

    calibration = scenario.calib
    spend = 0.0
    prev_obs: RoundObservation | None = None
    logs: list[dict] = []
    calib_logs: list[dict] = []
    exposure: dict[str, float] = {i: 0.0 for i in world._order}
    window_cost = 0.0

    for round_t in range(scenario.max_rounds):
        weights = scheduler.select_mixture(prev_obs)
        weights = apply_scenario_constraints(weights, world, round_t)
        obs = world.step_round(weights, steps=(round_t + 1) * scenario.steps_per_round)
        spend += obs.round_cost_usd
        window_cost += obs.round_cost_usd
        for env_id, w in weights.items():
            exposure[env_id] = exposure.get(env_id, 0.0) + w * world.M
        obs = replace(obs, budget_remaining_usd=max(world.budget_usd - spend, 0.0))
        scheduler.update_observation(obs)
        prev_obs = obs
        logs.append(
            {
                "round": obs.round,
                "weights": dict(weights),
                "round_cost_usd": obs.round_cost_usd,
                "spend_usd": spend,
                "per_env": {
                    env_id: {
                        "n_prompts": e.n_prompts,
                        "n_rollouts": e.n_rollouts,
                        "k_success": e.k_success,
                        "cost_usd": e.cost_usd,
                    }
                    for env_id, e in obs.per_env.items()
                },
                # diagnostics (never visible to the scheduler, Roadmap F.1)
                "true_skills": dict(world.skills),
                "true_pass_rates": world.true_pass_rates(),
            }
        )
        if (
            calibration.enabled
            and calibration.interval_rounds > 0
            and (round_t + 1) % calibration.interval_rounds == 0
        ):
            calib_obs: CalibrationObservation = world.calibration_observation(
                window_k=len(calib_logs) + 1,
                exposure_by_env=dict(exposure),
                window_cost_usd=window_cost,
                eval_cost_usd=calibration.cost_usd,
            )
            spend += calibration.cost_usd
            scheduler.update_calibration(calib_obs)
            calib_logs.append(
                {
                    "window_k": calib_obs.window_k,
                    "round": calib_obs.round,
                    "score_total": calib_obs.score_total,
                    "se_total": calib_obs.se_total,
                    "eval_cost_usd": calib_obs.eval_cost_usd,
                    "exposure_by_env": dict(calib_obs.exposure_by_env),
                }
            )
            exposure = {i: 0.0 for i in world._order}
            window_cost = 0.0
        if spend >= world.budget_usd:
            break

    final_score, final_se, _, _ = world.evaluate_benchmark(observed=True)  # uncharged
    return EpisodeResult(
        scenario_id=scenario.scenario_id,
        method=method,
        seed=seed,
        rounds=world.rounds_done,
        final_score=final_score,
        final_score_se=final_se,
        total_cost_usd=spend,
        budget_usd=world.budget_usd,
        benchmark_evals=world.benchmark_evals,
        round_logs=logs,
        calib_logs=calib_logs,
    )


def oracle_gap_closure(
    score_method: float, score_uniform: float, score_oracle: float
) -> float:
    """(S_method - S_uniform) / (S_oracle - S_uniform) at equal budget (F.4/F.5)."""
    denom = score_oracle - score_uniform
    if abs(denom) < 1e-12:
        raise ValueError("oracle and uniform scores coincide; scenario cannot discriminate")
    return (score_method - score_uniform) / denom


def mean_or(values: list[float], default: float = float("nan")) -> float:
    return sum(values) / len(values) if values else default


def scores_summary(scores: list[float]) -> dict[str, float]:
    n = len(scores)
    if n == 0:
        return {"mean": math.nan, "std": math.nan, "n": 0}
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / max(n - 1, 1)
    return {"mean": mean, "std": math.sqrt(var), "n": n}
