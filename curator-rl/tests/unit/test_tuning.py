"""Unit tests for the fair-tuning machinery (D-55, D-68)."""

from __future__ import annotations

import pytest
from experiments.sweeps.tune_all import GRIDS, TUNING_SCENARIOS, select, variant_cfg
from tests.conftest import BASE_CFG

from curator_rl.core.config import load_config


def _p(index, mean, se, ent):
    return {"index": index, "point": {"i": index}, "objective_mean": mean,
            "objective_se": se, "norm_entropy": ent}


def test_selection_picks_least_concentrated_among_one_se_candidates():
    # best = #1 (0.080, se 0.010) -> cutoff 0.070; #0 (0.075) and #2 (0.071) are candidates,
    # #3 (0.069) is not. Highest entropy among candidates wins: #2.
    points = [_p(0, 0.075, 0.01, 0.50), _p(1, 0.080, 0.01, 0.40),
              _p(2, 0.071, 0.01, 0.80), _p(3, 0.069, 0.01, 0.99)]
    sel = select(points)
    assert sel["best_index"] == 1
    assert sel["candidate_indices"] == [0, 1, 2]
    assert sel["chosen_index"] == 2


def test_selection_ties_on_entropy_go_to_best_objective():
    points = [_p(0, 0.070, 0.02, 0.6), _p(1, 0.075, 0.02, 0.6)]
    assert select(points)["chosen_index"] == 1


def test_selection_single_dominant_config():
    points = [_p(0, 0.10, 0.001, 0.2), _p(1, 0.05, 0.001, 0.9)]
    assert select(points)["chosen_index"] == 0     # 1 is far outside one SE of the best


def test_every_adaptive_method_has_the_same_tuning_budget():
    sizes = {m: len(g) for m, g in GRIDS.items()}
    assert set(sizes.values()) == {8}, sizes


def test_homogeneous_twin_is_held_out_of_tuning():
    assert "S-J" not in TUNING_SCENARIOS


def test_variant_cfg_changes_only_the_named_method():
    base = load_config(BASE_CFG)
    sec = variant_cfg(base, "sec", {"alpha": 0.9})
    assert sec.baselines.sec.alpha == pytest.approx(0.9)
    assert sec.baselines.dump == base.baselines.dump and sec.scheduler == base.scheduler
    cur = variant_cfg(base, "curator", {"gamma": 0.8, "tau": 2.0})
    assert (cur.scheduler.gamma, cur.scheduler.tau) == (0.8, 2.0)
    assert cur.baselines == base.baselines
    assert base.baselines.sec.alpha != 0.9                      # the base config is not mutated
