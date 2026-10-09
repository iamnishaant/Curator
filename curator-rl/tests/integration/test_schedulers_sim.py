"""Integration: Phase 5 schedulers over real simulator episodes.

Small-scale Gate-2 smoke: Curator must beat Uniform on S-A over 5 seeds, all
methods must respect the budget rule, and the decision records must be
well-formed (quotas sum to M, intents present).
"""

from __future__ import annotations

import pytest
from tests.conftest import REPO_ROOT

from curator_rl.simulator.harness import run_episode
from curator_rl.simulator.scenarios import load_scenario

SA = str(REPO_ROOT / "configs" / "sim" / "scenario_sa.yaml")


@pytest.mark.parametrize(
    "method", ["uniform", "static", "lp", "ucb", "curator"], ids=str
)
def test_all_methods_run_end_to_end_on_sa(method):
    scenario = load_scenario(SA)
    scenario = scenario.model_copy(update={"budget_usd": min(scenario.budget_usd, 1.0)})
    from experiments.run_sim import (
        make_scheduler,  # noqa: PLC0415  (repo-root on path via conftest)
    )

    scheduler = make_scheduler(method, scenario, seed=7)
    result = run_episode(scheduler, scenario, seed=7, method=method)
    assert result.within_budget_tolerance
    assert result.rounds > 0
    for log in result.round_logs:
        assert abs(sum(log["weights"].values()) - 1.0) < 1e-6
        assert all(w >= -1e-12 for w in log["weights"].values())


def test_curator_beats_uniform_on_sa_five_seeds():
    """Small-scale Gate-2 criterion (a): paired Curator - Uniform > 0 on S-A."""
    scenario = load_scenario(SA)
    scores = {}
    for method in ("uniform", "curator"):
        from experiments.run_sim import make_scheduler  # noqa: PLC0415

        runs = []
        for seed in range(5):
            scheduler = make_scheduler(method, scenario, seed=seed)
            result = run_episode(scheduler, scenario, seed=seed, method=method)
            runs.append(result.final_score)
        scores[method] = runs
    diffs = [c - u for c, u in zip(scores["curator"], scores["uniform"])]
    assert sum(diffs) / len(diffs) > 0.0, f"curator {scores['curator']} vs uniform {scores['uniform']}"


def test_curator_drops_the_saturated_arm():
    """The saturated arm (easy) loses share vs uniform; the noisy arm may keep
    weight pre-calibration (H3: that is what Phase 9 calibration is for)."""
    scenario = load_scenario(SA)
    from experiments.run_sim import make_scheduler  # noqa: PLC0415

    shares = {}
    for method in ("uniform", "curator"):
        scheduler = make_scheduler(method, scenario, seed=3)
        result = run_episode(scheduler, scenario, seed=3, method=method)
        late = [log for log in result.round_logs if log["round"] > 15]
        assert late
        shares[method] = {e: sum(log["weights"][e] for log in late) / len(late) for e in ("easy", "noisy", "valuable")}
    assert shares["curator"]["easy"] < shares["uniform"]["easy"]
    assert shares["curator"]["easy"] < 0.33
    # noisy share recorded as the H3 diagnostic (not asserted pre-calibration)
    print("noisy share (H3 diagnostic):", shares["curator"]["noisy"])
