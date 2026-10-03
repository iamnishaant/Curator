"""Per-environment contract suite (Roadmap D.5, Gate 1).

1. Determinism: same rng state -> identical prompts.
2. Gold answers verify >= 99%; garbage and empty strings fail >= 99%.
3. Timeout / bounded evaluation: oversized expressions are rejected quickly.
4. Split membership: a prompt served for a split belongs to that split; the
   sealed split is never served by the environment.
5. Noisy success rate within 0.02 of its design value.

Real-data tests (GSM8K, noisy) skip when splits have not been built.
"""

import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from curator_rl.core.config import load_config
from curator_rl.envs.countdown import CountdownEnv, CountdownError, safe_arith_eval

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"


@pytest.fixture(scope="module")
def cfg():
    return load_config(REPO_ROOT / "configs" / "base.yaml")


@pytest.fixture(scope="module")
def envs(cfg):
    from curator_rl.envs.gsm8k import Gsm8kEnv
    from curator_rl.envs.noisy import NoisyRewardEnv

    built = (DATA_ROOT / "processed" / "gsm8k" / "train.jsonl").exists()
    return {
        "gsm8k": Gsm8kEnv(REPO_ROOT, cfg.data.envs.gsm8k) if built else None,
        "countdown": CountdownEnv(REPO_ROOT, cfg.data.envs.countdown),
        "noisy": NoisyRewardEnv(REPO_ROOT, cfg.data.envs.noisy) if built else None,
    }


DATA_ENVS = ["gsm8k", "noisy"]


def _require_env(envs, name):
    env = envs[name]
    if env is None:
        pytest.skip("GSM8K splits not built (run scripts/download_data.py && scripts/build_splits.py)")
    return env


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------


def test_determinism_same_rng_state_all_envs(envs):
    for name in ("gsm8k", "countdown", "noisy"):
        env = _require_env(envs, name)
        a = env.generate_prompt("train", np.random.default_rng(1234))
        b = env.generate_prompt("train", np.random.default_rng(1234))
        assert a.prompt_id == b.prompt_id
        assert a.messages == b.messages
        assert a.reference == b.reference
        assert a.meta == b.meta


def test_different_seeds_differ_somewhere(envs):
    for name in ("gsm8k", "countdown", "noisy"):
        env = _require_env(envs, name)
        seen = {env.generate_prompt("train", np.random.default_rng(s)).prompt_id for s in range(20)}
        assert len(seen) > 1


# ---------------------------------------------------------------------------
# Batch / sampling without replacement
# ---------------------------------------------------------------------------


def test_batch_without_replacement(envs):
    for name in ("gsm8k", "countdown", "noisy"):
        env = _require_env(envs, name)
        batch = env.generate_batch("train", 12, np.random.default_rng(7))
        ids = [p.prompt_id for p in batch]
        assert len(ids) == len(set(ids)) == 12


# ---------------------------------------------------------------------------
# 2. Gold / garbage verdicts
# ---------------------------------------------------------------------------


GARBAGE_RESPONSES = ["", "no idea", "\\boxed{}", "\\boxed{banana}", "\\boxed[, ]"]


def test_gsm8k_gold_and_garbage(envs):
    env = _require_env(envs, "gsm8k")
    rng = np.random.default_rng(99)
    prompts = env.generate_batch("train", 60, rng)
    gold_ok = sum(env.evaluate_response(p, f"working... \\boxed{{{p.reference}}}").success for p in prompts)
    assert gold_ok / len(prompts) >= 0.99
    garbage_fail = sum(
        not env.evaluate_response(p, g).success for p in prompts for g in GARBAGE_RESPONSES
    )
    assert garbage_fail >= 0.99 * len(prompts) * len(GARBAGE_RESPONSES)


def test_gsm8k_numeric_normalisation(envs):
    env = _require_env(envs, "gsm8k")
    p = env.generate_prompt("train", np.random.default_rng(3))
    ref = int(p.reference) if re.fullmatch(r"-?\d+", p.reference) else None
    if ref is None:
        pytest.skip("sampled reference is not a plain integer")
    for variant in (str(ref), f"{ref}.0", f"${ref:,}", f" {ref} "):
        assert env.evaluate_response(p, f"\\boxed{{{variant}}}").success


def test_countdown_gold_and_garbage(envs):
    env = envs["countdown"]
    rng = np.random.default_rng(11)
    prompts = env.generate_batch("train", 60, rng)
    gold_ok = sum(env.evaluate_response(p, f"\\boxed{{{p.meta['solution']}}}").success for p in prompts)
    assert gold_ok / len(prompts) >= 0.99
    garbage_fail = sum(
        not env.evaluate_response(p, g).success for p in prompts[:10] for g in GARBAGE_RESPONSES
    )
    assert garbage_fail >= 0.99 * 10 * len(GARBAGE_RESPONSES)


def test_noisy_evaluation_is_honest_but_reward_is_noise(envs, cfg):
    env = _require_env(envs, "noisy")
    rng = np.random.default_rng(5)
    prompts = env.generate_batch("train", 40, rng)
    gold_ok = sum(env.evaluate_response(p, f"\\boxed{{{p.reference}}}").success for p in prompts)
    assert gold_ok / len(prompts) >= 0.99  # verification unchanged

    # reward rate ~= q for BOTH gold and garbage (independent of outcome).
    # The response text varies per draw so each rollout gets its own hash
    # draw - a fixed response would give one fixed Bernoulli outcome.
    q = cfg.data.envs.noisy.q
    n = 4000
    rewards_gold = 0
    rewards_garbage = 0
    p = prompts[0]
    for i in range(n):
        v_gold = env.evaluate_response(p, f"reasoning step {i}. \\boxed{{{p.reference}}}")
        v_junk = env.evaluate_response(p, f"filler text {i} \\boxed{{banana}}")
        rewards_gold += env.compute_reward(v_gold)
        rewards_garbage += env.compute_reward(v_junk)
    assert abs(rewards_gold / n - q) <= 0.02
    assert abs(rewards_garbage / n - q) <= 0.02


# ---------------------------------------------------------------------------
# 3. Timeout / bounded evaluation
# ---------------------------------------------------------------------------


def test_countdown_bounded_evaluation(envs):
    env = envs["countdown"]
    prompt = env.generate_prompt("train", np.random.default_rng(1))
    huge = "\\boxed{" + "(" * 100 + "1" + ")" * 100 + "}"
    verdict = env.evaluate_response(prompt, huge)
    assert not verdict.success
    # the verifier itself stays bounded: well under the configured timeout
    assert verdict.verifier_seconds < env.cfg.verifier_timeout_s


# ---------------------------------------------------------------------------
# 4. Split membership and sealing
# ---------------------------------------------------------------------------


def test_served_prompt_belongs_to_requested_split(envs):
    for name in ("gsm8k", "countdown", "noisy"):
        env = _require_env(envs, name)
        for split in ("train", "calib", "dev"):
            if name == "noisy" and split != "train":
                continue  # noisy only has a train partition
            prompt = env.generate_prompt(split, np.random.default_rng(0))
            assert prompt.split == split


def test_sealed_split_never_served(envs):
    for name in ("gsm8k", "noisy"):
        env = _require_env(envs, name)
        with pytest.raises(ValueError, match="sealed"):
            env.generate_prompt("test", np.random.default_rng(0))


def test_countdown_unknown_split_rejected(envs):
    # countdown CAN serve 'test' (procedural instances from the sealed seed
    # range, D-22); an unknown split must raise.
    with pytest.raises(ValueError, match="unknown split"):
        envs["countdown"].generate_prompt("val", np.random.default_rng(0))


def test_gsm8k_split_ids_match_manifest(envs):
    from curator_rl.envs.splits import read_manifest_ids

    env = _require_env(envs, "gsm8k")
    for split in ("train", "calib", "dev"):
        served = set(env.split_ids(split))
        assert served == set(read_manifest_ids(DATA_ROOT, "gsm8k", split))


def test_countdown_seed_ranges_disjoint_by_design(cfg):
    from curator_rl.envs.countdown import SPLIT_SEED_RANGES

    ends = sorted(hi for _, hi in SPLIT_SEED_RANGES.values())
    starts = sorted(lo for lo, _ in SPLIT_SEED_RANGES.values())
    for end, start in zip(ends[:-1], starts[1:], strict=True):
        assert end <= start


# ---------------------------------------------------------------------------
# safe_arith_eval: the safety-critical evaluator
# ---------------------------------------------------------------------------


class TestSafeArithEval:
    def test_basic_exact_arithmetic(self):
        assert safe_arith_eval("(1+2)*3", [1, 2, 3], require_all=True) == Fraction(9)
        assert safe_arith_eval("6/2", [6, 2], require_all=True) == Fraction(3)
        assert safe_arith_eval("1/3", [1, 3], require_all=True) == Fraction(1, 3)
        assert safe_arith_eval("1/3*3", [1, 3, 3], require_all=True) == Fraction(1)

    def test_rejects_unused_numbers(self):
        with pytest.raises(CountdownError, match="unused numbers"):
            safe_arith_eval("1+2", [1, 2, 3], require_all=True)

    def test_rejects_repeating_a_number(self):
        with pytest.raises(CountdownError, match="not available"):
            safe_arith_eval("2*2", [2, 3], require_all=True)

    def test_rejects_banned_nodes(self):
        with pytest.raises(CountdownError):
            safe_arith_eval("__import__('os')", [1, 2], require_all=False)
        with pytest.raises(CountdownError):
            safe_arith_eval("(1+2)(3)", [1, 2, 3], require_all=True)
        with pytest.raises(CountdownError):
            safe_arith_eval("1.5+2", [1, 2], require_all=True)
        with pytest.raises(CountdownError):
            safe_arith_eval("'a'+'b'", [1], require_all=True)
        with pytest.raises(CountdownError):
            safe_arith_eval("1 if 2 else 3", [1, 2, 3], require_all=True)

    def test_rejects_division_by_zero(self):
        with pytest.raises(CountdownError, match="division by zero"):
            safe_arith_eval("1/0", [1, 0], require_all=True)

    def test_negative_unary(self):
        assert safe_arith_eval("-1+3", [1, 3], require_all=True) == Fraction(2)

    def test_oversize_expression_rejected(self):
        with pytest.raises(CountdownError, match="too long"):
            safe_arith_eval("1+" * 200 + "1", [1], require_all=False)
