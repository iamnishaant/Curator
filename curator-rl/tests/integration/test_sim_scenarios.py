"""Integration tests: every scenario runs end to end (Roadmap F.5, Phase 3 item 11).

Budgets are shrunk (not the committed scenario values) to keep the suite CPU-
cheap; the full-budget scenario validation with oracles is run through
`experiments/run_sim.py` and its reports.
"""

from pathlib import Path

import pytest

from curator_rl.core.seeding import SeedManager
from curator_rl.scheduler.baselines.random_baseline import RandomScheduler
from curator_rl.scheduler.baselines.uniform import UniformScheduler
from curator_rl.simulator.harness import run_episode
from curator_rl.simulator.oracle import StaticOracle
from curator_rl.simulator.scenarios import load_scenario

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "configs" / "sim"
SCENARIOS = sorted(SCENARIO_DIR.glob("scenario_s*.yaml"))


def shrunk(scenario, budget_usd=0.6, max_rounds=60):
    return scenario.model_copy(update={"budget_usd": budget_usd, "max_rounds": max_rounds})


def test_all_eight_scenarios_present():
    assert len(SCENARIOS) == 8, f"expected S-A..S-H, found {[p.name for p in SCENARIOS]}"


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_scenario_runs_end_to_end(path):
    scenario = shrunk(load_scenario(path))
    for method, make in (
        ("uniform", lambda envs: UniformScheduler(envs)),
        ("random", lambda envs: RandomScheduler(envs, SeedManager(1).rng("scheduler"))),
    ):
        result = run_episode(make([e.env_id for e in scenario.envs]), scenario, seed=1,
                             method=method)
        assert result.rounds > 0
        assert 0.0 <= result.final_score <= 1.0
        assert result.within_budget_tolerance, (
            f"{scenario.scenario_id}/{method}: cost {result.total_cost_usd:.3f} "
            f"vs budget {result.budget_usd:.3f}"
        )
        assert len(result.round_logs) == result.rounds
        for log in result.round_logs:
            assert abs(sum(log["weights"].values()) - 1.0) < 1e-6


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_scenario_determinism_by_seed(path):
    scenario = shrunk(load_scenario(path), budget_usd=0.3, max_rounds=30)
    env_ids = [e.env_id for e in scenario.envs]
    r1 = run_episode(UniformScheduler(env_ids), scenario, seed=42, method="uniform")
    r2 = run_episode(UniformScheduler(env_ids), scenario, seed=42, method="uniform")
    assert r1.final_score == r2.final_score
    assert r1.total_cost_usd == r2.total_cost_usd
    assert r1.rounds == r2.rounds


def test_calibration_produced_and_charged():
    scenario = load_scenario(SCENARIO_DIR / "scenario_sg.yaml")  # calib enabled
    scenario = shrunk(scenario, budget_usd=0.5, max_rounds=60)
    env_ids = [e.env_id for e in scenario.envs]
    result = run_episode(UniformScheduler(env_ids), scenario, seed=3, method="uniform")
    assert len(result.calib_logs) > 0
    assert all(log["eval_cost_usd"] == scenario.calib.cost_usd for log in result.calib_logs)


def test_scenario_sa_oracle_gap_above_ten_percent():
    """Exit criterion (Phase 3 item 12): oracle beats uniform by > 10% in S-A."""
    scenario = load_scenario(SCENARIO_DIR / "scenario_sa.yaml")
    env_ids = [e.env_id for e in scenario.envs]
    uniform_scores = []
    oracle_scores = []
    for seed in range(5):
        uniform_scores.append(
            run_episode(UniformScheduler(env_ids), scenario, seed=seed, method="uniform").final_score
        )
        oracle_scores.append(
            run_episode(StaticOracle(_world_of(scenario, seed), simplex_step=0.1, seed=seed),
                        scenario, seed=seed, method="static_oracle").final_score
        )
    diff = sum(oracle_scores) / 5 - sum(uniform_scores) / 5
    assert diff > 0.10, f"oracle improvement {diff:.3f} not above 10 points"


def _world_of(scenario, seed: int):
    from curator_rl.simulator.scenarios import build_world

    return build_world(scenario, seed)
