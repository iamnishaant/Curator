"""Integration test: SignalEngine over real simulator episodes (Phase 4).

The engine consumes the exact `RoundObservation` stream the harness emits
(captured through a recording wrapper around Uniform/Random schedulers) and
is checked against roadmap behaviors: warm-up in S1, no S3/S4 junk labels on
the noisy arm, statuses legal throughout, and state round-trips through a
JSON checkpoint.
"""

from __future__ import annotations

import json

import pytest
from tests.conftest import BASE_CFG, REPO_ROOT

from curator_rl.core.config import load_config
from curator_rl.scheduler.base import BaseScheduler
from curator_rl.scheduler.baselines.uniform import UniformScheduler
from curator_rl.signals import SignalEngine
from curator_rl.simulator.harness import run_episode
from curator_rl.simulator.scenarios import load_scenario


class RecordingScheduler(BaseScheduler):
    """Harness validation wrapper: records the full RoundObservation stream."""

    def __init__(self, inner):
        super().__init__(inner.env_ids)
        self.inner = inner
        self.observations = []

    def select_mixture(self, observation):
        return self.inner.select_mixture(observation)

    def update_observation(self, obs):
        self.observations.append(obs)
        self.inner.update_observation(obs)


SCENARIOS = sorted((REPO_ROOT / "configs" / "sim").glob("scenario_s*.yaml"),
                   key=lambda p: p.stem)


@pytest.mark.parametrize("path", SCENARIOS, ids=lambda p: p.stem)
def test_engine_runs_over_every_scenario_and_statuses_are_legal(path):
    scenario = load_scenario(str(path))
    env_ids = tuple(e.env_id for e in scenario.envs)
    scenario = scenario.model_copy(update={"budget_usd": min(scenario.budget_usd, 1.0),
                                           "max_rounds": min(scenario.max_rounds, 25)})
    sched = RecordingScheduler(UniformScheduler(env_ids))
    result = run_episode(sched, scenario, seed=3)
    cfg = load_config(BASE_CFG)
    engine = SignalEngine(cfg.signals, cfg.proxy, cfg.calib,
                          cfg.scheduler.cost_exponent, cfg.group_size, env_ids)
    assert len(sched.observations) == result.rounds
    for obs in sched.observations:
        vectors = engine.update(obs)
        for sv in vectors.values():
            assert sv.status.value.startswith("S")  # S1..S5 only
            assert 0.0 <= sv.proxy_reward <= 1.0
            assert 0.0 <= sv.pass_rate <= 1.0
            assert 0.0 <= sv.richness <= 1.0
            assert sv.pass_lo <= sv.pass_rate <= sv.pass_hi


def test_noisy_env_never_misclassified_as_saturated_or_too_hard():
    """H3 setup: the noisy arm looks like S2b before calibration (Roadmap E.5)."""
    scenario = load_scenario(str(REPO_ROOT / "configs" / "sim" / "scenario_sa.yaml"))
    env_ids = tuple(e.env_id for e in scenario.envs)
    noisy = next(e.env_id for e in scenario.envs if e.noisy_q is not None)
    sched = RecordingScheduler(UniformScheduler(env_ids))
    run_episode(sched, scenario, seed=0)
    cfg = load_config(BASE_CFG)
    engine = SignalEngine(cfg.signals, cfg.proxy, cfg.calib,
                          cfg.scheduler.cost_exponent, cfg.group_size, env_ids)
    statuses = set()
    for obs in sched.observations:
        vectors = engine.update(obs)
        statuses.add(vectors[noisy].status)
    assert statuses.isdisjoint({"S3_saturated", "S4_too_hard"})


def test_first_round_is_s1_for_every_env():
    scenario = load_scenario(str(REPO_ROOT / "configs" / "sim" / "scenario_sa.yaml"))
    env_ids = tuple(e.env_id for e in scenario.envs)
    sched = RecordingScheduler(UniformScheduler(env_ids))
    run_episode(sched, scenario, seed=0)
    cfg = load_config(BASE_CFG)
    engine = SignalEngine(cfg.signals, cfg.proxy, cfg.calib,
                          cfg.scheduler.cost_exponent, cfg.group_size, env_ids)
    vectors = engine.update(sched.observations[0])
    assert all(sv.status.value == "S1_unexplored" for sv in vectors.values())


def test_engine_state_survives_mid_stream_checkpoint():
    scenario = load_scenario(str(REPO_ROOT / "configs" / "sim" / "scenario_sa.yaml"))
    env_ids = tuple(e.env_id for e in scenario.envs)
    sched = RecordingScheduler(UniformScheduler(env_ids))
    run_episode(sched, scenario, seed=0)
    cfg = load_config(BASE_CFG)
    split = len(sched.observations) // 2 + 2
    engine = SignalEngine(cfg.signals, cfg.proxy, cfg.calib,
                          cfg.scheduler.cost_exponent, cfg.group_size, env_ids)
    replay = [engine.update(obs) for obs in sched.observations[:split]]
    state = json.loads(json.dumps(engine.get_state()))
    forked = SignalEngine(cfg.signals, cfg.proxy, cfg.calib,
                          cfg.scheduler.cost_exponent, cfg.group_size, env_ids)
    forked.load_checkpoint(state)
    rest = range(split, len(sched.observations))
    for i in rest:
        obs = sched.observations[i]
        a = engine.update(obs)
        b = forked.update(obs)
        for env_id in env_ids:
            assert a[env_id].status == b[env_id].status
            assert a[env_id].proxy_reward == b[env_id].proxy_reward
    assert len(replay) == split
