"""Unit tests for the synthetic world (Roadmap F.2, v1 Phase 3 item 10)."""

import numpy as np

from curator_rl.simulator.world import SimEnvParams, SimWorld, learnability, sigmoid


def make_world(envs: list[SimEnvParams], seed: int = 0, **kwargs) -> SimWorld:
    defaults = dict(
        steps_per_round=5,
        prompts_per_round=80,
        group_size=8,
        budget_usd=2.0,
        skill_noise_std=0.0,
        cost_lognormal_sigma=0.0,
    )
    defaults.update(kwargs)
    return SimWorld(envs, **defaults, rng=np.random.default_rng(seed))


def test_pass_rate_in_unit_interval():
    e = SimEnvParams(env_id="a", slope=3.0, difficulty=0.2, eta=0.01,
                     cost_usd_per_prompt=0.001)
    world = make_world([e])
    for delta in (-5.0, -1.0, 0.0, 0.5, 2.0, 10.0):
        world.skills["a"] = delta
        p = world.true_pass("a")
        assert 0.0 < p < 1.0


def test_learnability_peaks_at_half():
    assert learnability(0.5) == 1.0
    assert learnability(0.5) > learnability(0.2)
    assert learnability(0.5) > learnability(0.8)
    assert learnability(0.0) == 0.0
    assert learnability(1.0) == 0.0


def test_zero_transfer_env_does_not_affect_others():
    driver = SimEnvParams(env_id="driver", difficulty=0.0, eta=0.02,
                          cost_usd_per_prompt=0.001, transfer_out={})
    other = SimEnvParams(env_id="other", difficulty=0.0, eta=0.02,
                         cost_usd_per_prompt=0.001)
    world = make_world([driver, other])
    before = world.skills["other"]
    for _ in range(10):
        obs = world.step_round({"driver": 1.0, "other": 0.0})
    assert obs.per_env["driver"].n_prompts == 80
    assert world.skills["driver"] > before
    assert world.skills["other"] == before  # zero transfer, zero allocation


def test_noisy_env_skill_frozen():
    normal = SimEnvParams(env_id="normal", difficulty=0.0, eta=0.02,
                          cost_usd_per_prompt=0.001)
    noisy = SimEnvParams(env_id="noisy", eta=0.02, cost_usd_per_prompt=0.001,
                         noisy_q=0.3)
    world = make_world([normal, noisy], skill_noise_std=0.01)
    for _ in range(10):
        world.step_round({"normal": 0.0, "noisy": 1.0})
    assert world.skills["noisy"] == 0.0
    # noisy rewards are independent of skill but in expectation hit q
    assert abs(world.true_pass("noisy") - 0.5) < 1e-9  # skill 0, d 0 => p 0.5


def test_noisy_reward_rate_hits_design_value():
    noisy = SimEnvParams(env_id="noisy", eta=0.01, cost_usd_per_prompt=0.001,
                         noisy_q=0.35)
    world = make_world([noisy], skill_noise_std=0.0)
    total_k = 0
    total_n = 0
    for _ in range(200):
        o = world.step_round({"noisy": 1.0})
        total_k += o.per_env["noisy"].k_success
        total_n += o.per_env["noisy"].n_rollouts
    rate = total_k / total_n
    assert abs(rate - 0.35) < 0.02  # design value within 0.02 (Roadmap D.5 #5)


def test_round_cost_matches_weighted_mean_cost():
    a = SimEnvParams(env_id="a", difficulty=0.0, eta=0.01, cost_usd_per_prompt=0.001)
    b = SimEnvParams(env_id="b", difficulty=0.0, eta=0.01, cost_usd_per_prompt=0.003)
    world = make_world([a, b])  # cost_lognormal_sigma=0 => deterministic costs
    obs = world.step_round({"a": 0.25, "b": 0.75})
    expected = 80 * (0.25 * 0.001 + 0.75 * 0.003)
    assert abs(obs.round_cost_usd - expected) < 1e-9


def test_quota_rounding_sums_to_M():
    a = SimEnvParams(env_id="a", difficulty=0.0, eta=0.01, cost_usd_per_prompt=0.001)
    b = SimEnvParams(env_id="b", difficulty=0.0, eta=0.01, cost_usd_per_prompt=0.001)
    c = SimEnvParams(env_id="c", difficulty=0.0, eta=0.01, cost_usd_per_prompt=0.001)
    world = make_world([a, b, c])
    obs = world.step_round({"a": 0.2, "b": 0.3, "c": 0.5})
    assert sum(o.n_prompts for o in obs.per_env.values()) == 80
    assert obs.per_env["a"].n_prompts == 16
    assert obs.per_env["b"].n_prompts == 24
    assert obs.per_env["c"].n_prompts == 40


def test_determinism_by_seed():
    envs = [
        SimEnvParams(env_id="a", difficulty=0.3, eta=0.012, cost_usd_per_prompt=0.001),
        SimEnvParams(env_id="b", difficulty=0.7, eta=0.01, cost_usd_per_prompt=0.002,
                     noisy_q=0.3),
    ]
    runs = []
    for _ in range(2):
        world = make_world(envs, seed=123, skill_noise_std=0.003,
                           cost_lognormal_sigma=0.1)
        scores = []
        for _ in range(5):
            obs = world.step_round({"a": 0.6, "b": 0.4})
            scores.append(
                (obs.round_cost_usd,
                 obs.per_env["a"].k_success,
                 obs.per_env["b"].k_success,
                 round(world.skills["a"], 9),
                 round(world.skills["b"], 9))
            )
        runs.append(scores)
    assert runs[0] == runs[1]


def test_benchmark_score_within_unit_interval_and_weights():
    a = SimEnvParams(env_id="a", difficulty=0.0, eta=0.01, cost_usd_per_prompt=0.001,
                     bench_weight=0.3)
    b = SimEnvParams(env_id="b", difficulty=1.0, eta=0.01, cost_usd_per_prompt=0.001,
                     bench_weight=0.7)
    world = make_world([a, b])
    assert abs(sum(world.bench_weight.values()) - 1.0) < 1e-9
    s, se, by_domain, se_by_domain = world.evaluate_benchmark(observed=True)
    assert 0.0 <= s <= 1.0
    assert 0.0 < se <= 0.5
    assert set(by_domain) == {"a", "b"}
    expected_true = world.expected_benchmark_score()
    assert 0.0 <= expected_true <= 1.0


def test_drift_shifts_difficulty_once():
    e = SimEnvParams(env_id="a", difficulty=0.0, eta=0.01, cost_usd_per_prompt=0.001,
                     drift_round=2, drift_shift=-1.0)
    world = make_world([e])
    p_before = world.true_pass("a")
    world.step_round({"a": 1.0})  # round 1: no shift yet
    assert world.envs["a"].difficulty == 0.0
    world.step_round({"a": 1.0})  # round 2 completes -> shift applies
    assert world.envs["a"].difficulty == -1.0
    assert world.true_pass("a") > p_before


def test_gate_blocks_learning_until_prereq_ready():
    prereq = SimEnvParams(env_id="pre", difficulty=2.0, eta=0.05,
                          cost_usd_per_prompt=0.001)
    gated = SimEnvParams(env_id="gated", difficulty=0.0, eta=0.05,
                         cost_usd_per_prompt=0.001, gate_prereq="pre",
                         gate_theta=1.0, gate_slope=6.0)
    world = make_world([prereq, gated], skill_noise_std=0.0)
    for _ in range(20):
        world.step_round({"pre": 1.0, "gated": 0.0})
    assert world.skills["gated"] == 0.0
    # prereq skill 20 * 5 * 0.05 * g(0.5) = 2.5 > theta 1.0 -> gate open
    world.step_round({"pre": 0.0, "gated": 1.0})
    assert world.skills["gated"] > 0.0


def test_sigmoid_matches_numpy():
    import numpy as np

    for x in (-4.0, -0.5, 0.0, 0.5, 4.0):
        assert abs(sigmoid(x) - 1.0 / (1.0 + np.exp(-x))) < 1e-12
