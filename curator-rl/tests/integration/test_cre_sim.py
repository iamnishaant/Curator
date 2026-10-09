"""End-to-end tests of the Calibrated Reward Engine inside Curator (D-76)."""

from __future__ import annotations

import ast

import pytest
from experiments.run_sim import make_scheduler
from tests.conftest import REPO_ROOT

from curator_rl.core.config import load_config
from curator_rl.simulator.harness import run_episode
from curator_rl.simulator.scenarios import CalibSimCfg, load_scenario

SCEN = REPO_ROOT / "configs" / "sim"


def cfg_with(cre_enabled=True, **cre_kwargs):
    base = load_config(REPO_ROOT / "configs" / "base.yaml")
    cre = base.cre.model_copy(update={"enabled": cre_enabled, **cre_kwargs})
    return base.model_copy(update={"cre": cre})


def calibrated(scenario_file="scenario_si.yaml", budget=None):
    sc = load_scenario(SCEN / scenario_file)
    cost_item = sum(e.cost_usd_per_prompt for e in sc.envs) / len(sc.envs) / sc.group_size
    sc = sc.model_copy(update={"calib": CalibSimCfg(
        enabled=True, interval_rounds=10, paired=True, churn=0.02,
        items_per_slice=100, cost_per_item_usd=cost_item)})
    return sc if budget is None else sc.model_copy(update={"budget_usd": budget})


def test_calibration_feedback_changes_the_allocation_and_logs_every_reward():
    sc = calibrated()
    plain = make_scheduler("curator", sc, 3, cfg=cfg_with(cre_enabled=False))
    cre = make_scheduler("curator", sc, 3, cfg=cfg_with())
    rp = run_episode(plain, sc, seed=3, method="plain")
    rc = run_episode(cre, sc, seed=3, method="cre")
    sources = {v["src"] for row in cre.reward_log for k, v in row.items() if k != "round"}
    assert "proxy" in sources and any(s.startswith("cre:") for s in sources)   # switched on after calibration
    assert not any(v["src"].startswith("cre:") for row in plain.reward_log for k, v in row.items() if k != "round")
    assert [log["weights"] for log in rp.round_logs] != [log["weights"] for log in rc.round_logs]
    assert len(cre.reward_log) == rc.rounds                                    # every round is logged
    first_cre = next(i for i, row in enumerate(cre.reward_log)
                     if any(v["src"].startswith("cre:") for k, v in row.items() if k != "round"))
    for k, v in cre.reward_log[first_cre].items():
        if k != "round":
            assert {"r", "src", "b", "sd", "n"} <= set(v) and 0.0 <= v["r"] <= 1.0 and v["sd"] > 0


def test_weights_stay_valid_with_the_cre_active():
    sc = calibrated()
    cfg = cfg_with()
    res = run_episode(make_scheduler("curator", sc, 5, cfg=cfg), sc, seed=5, method="cre")
    floor = cfg.scheduler.epsilon / len(sc.envs)
    for log in res.round_logs:
        w = log["weights"]
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert all(v >= floor - 1e-9 for v in w.values())


def test_calibration_is_charged_exactly_once_and_only_to_its_user():
    sc = calibrated(budget=4.0)
    res = run_episode(make_scheduler("curator", sc, 2, cfg=cfg_with()), sc, seed=2, method="cre")
    train = sum(log["round_cost_usd"] for log in res.round_logs)
    calib = sum(c["eval_cost_usd"] for c in res.calib_logs)
    assert calib > 0
    assert res.total_cost_usd == pytest.approx(train + calib)
    for method in ("uniform", "ucb", "sec", "dump"):               # not users: never charged
        r = run_episode(make_scheduler(method, sc, 2, cfg=cfg_with()), sc, seed=2, method=method)
        assert r.calib_logs == []
        assert r.total_cost_usd == pytest.approx(sum(log["round_cost_usd"] for log in r.round_logs))


def test_cre_is_inert_when_the_scenario_offers_no_calibration():
    """Missing calibration: CRE never activates and the run equals Curator without CRE."""
    sc = load_scenario(SCEN / "scenario_si.yaml")            # calib disabled in the file
    a = make_scheduler("curator", sc, 4, cfg=cfg_with(cre_enabled=False))
    b = make_scheduler("curator", sc, 4, cfg=cfg_with())
    ra = run_episode(a, sc, seed=4, method="a")
    rb = run_episode(b, sc, seed=4, method="b")
    assert [x["weights"] for x in ra.round_logs] == [x["weights"] for x in rb.round_logs]
    assert not b._cre.active                                  # noqa: SLF001


def test_standard_ucb_never_uses_the_cre():
    sc = calibrated()
    ucb = make_scheduler("ucb", sc, 1, cfg=cfg_with())
    assert ucb._cre is None and ucb.uses_calibration is False  # noqa: SLF001


def test_runs_are_deterministic_and_checkpoint_state_roundtrips():
    sc = calibrated(budget=5.0)
    s1 = make_scheduler("curator", sc, 7, cfg=cfg_with())
    s2 = make_scheduler("curator", sc, 7, cfg=cfg_with())
    r1 = run_episode(s1, sc, seed=7, method="x")
    r2 = run_episode(s2, sc, seed=7, method="x")
    assert [x["weights"] for x in r1.round_logs] == [x["weights"] for x in r2.round_logs]
    fresh = make_scheduler("curator", sc, 7, cfg=cfg_with())
    fresh.load_checkpoint(s1.get_state())
    assert fresh.get_state()["cre"] == s1.get_state()["cre"]
    assert fresh.get_state()["calibrator"] == s1.get_state()["calibrator"]


def test_exact_cost_ablation_overrides_the_measured_cost():
    sc = calibrated(budget=5.0)
    sched = make_scheduler("curator", sc, 6, cfg=cfg_with())
    sched.cost_override = {e.env_id: e.cost_usd_per_prompt for e in sc.envs}
    res = run_episode(sched, sc, seed=6, method="exact")
    assert res.rounds > 0 and any(
        v["src"].startswith("cre:") for row in sched.reward_log for k, v in row.items() if k != "round")


def test_cre_module_cannot_see_hidden_simulator_state():
    """Leakage guard: the engine and calibrator import no simulator, env or evaluation code."""
    for rel in ("cre.py", "calibrator.py", "credit.py"):
        tree = ast.parse((REPO_ROOT / "src" / "curator_rl" / "calibration" / rel).read_text(encoding="utf-8"))
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert not any(m.startswith(("curator_rl.simulator", "curator_rl.envs", "curator_rl.evaluation"))
                       for m in mods), (rel, mods)
