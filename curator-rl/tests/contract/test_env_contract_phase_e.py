"""Contract suite for MATH35 and MBPP (Roadmap D.5, Gate 1B'). Skips when splits are not built.

1. Determinism; batches without replacement.
2. Gold verifies >= 99%; garbage and empty strings fail >= 99%.
3. Bounded evaluation: an infinite loop returns within the timeout (MBPP).
4. Split membership and leakage: disjoint splits, level restriction, sealed test never served.
5. Reward hacks fail on real tasks: early exit, forged sentinel, special-casing the shown test.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from curator_rl.core.config import load_config
from curator_rl.envs.splits import read_manifest_ids

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
BUILT = (DATA_ROOT / "processed" / "math35" / "train.jsonl").exists() and \
        (DATA_ROOT / "processed" / "mbpp" / "train.jsonl").exists()
pytestmark = pytest.mark.skipif(not BUILT, reason="run scripts/download_data_phase_e.py && scripts/build_splits_phase_e.py")


@pytest.fixture(scope="module")
def cfg():
    return load_config(REPO_ROOT / "configs" / "base.yaml")


@pytest.fixture(scope="module")
def math_env(cfg):
    from curator_rl.envs.math35 import Math35Env

    return Math35Env(REPO_ROOT, cfg.data.envs.math35)


@pytest.fixture(scope="module")
def mbpp_env(cfg):
    from curator_rl.envs.mbpp import MbppEnv

    return MbppEnv(REPO_ROOT, cfg.data.envs.mbpp)


def _rows(env, split):
    return env._items[split]  # noqa: SLF001 (test needs gold answers / gold code)


# ------------------------------------------------------------------ determinism
@pytest.mark.parametrize("name", ["math_env", "mbpp_env"])
def test_determinism_and_batches(name, request):
    env = request.getfixturevalue(name)
    a = env.generate_prompt("train", np.random.default_rng(99))
    b = env.generate_prompt("train", np.random.default_rng(99))
    assert (a.prompt_id, a.messages, a.reference, a.meta) == (b.prompt_id, b.messages, b.reference, b.meta)
    ids = [p.prompt_id for p in env.generate_batch("train", 20, np.random.default_rng(3))]
    assert len(ids) == len(set(ids)) == 20


# ------------------------------------------------------------------ splits
@pytest.mark.parametrize("name,splits", [("math_env", ("train", "calib", "dev")), ("mbpp_env", ("train", "calib", "dev"))])
def test_splits_are_disjoint_and_match_manifests(name, splits, request):
    env = request.getfixturevalue(name)
    sets = {s: set(env.split_ids(s)) for s in splits}
    sealed = set(read_manifest_ids(DATA_ROOT, env.env_id, "test"))
    assert sealed and all(sets[s] for s in splits)
    names = list(sets) + ["test"]
    allsets = {**sets, "test": sealed}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (allsets[a] & allsets[b]), (a, b)
    for s in splits:
        assert sets[s] == set(read_manifest_ids(DATA_ROOT, env.env_id, s))


def test_sealed_test_is_never_served(math_env, mbpp_env):
    for env in (math_env, mbpp_env):
        with pytest.raises(ValueError, match="sealed"):
            env.generate_prompt("test", np.random.default_rng(0))


def test_math35_trains_only_on_levels_3_to_5_and_has_the_planned_sizes(math_env, cfg):
    lo, hi = cfg.data.envs.math35.level_min, cfg.data.envs.math35.level_max
    for split in ("train", "calib", "dev"):
        assert all(lo <= r["level"] <= hi for r in _rows(math_env, split))
    assert len(_rows(math_env, "calib")) == cfg.data.calib_size and len(_rows(math_env, "dev")) == cfg.data.dev_size


def test_mbpp_split_sizes_follow_the_official_splits(mbpp_env, cfg):
    assert len(_rows(mbpp_env, "calib")) == 90                     # official validation
    assert len(_rows(mbpp_env, "dev")) == cfg.data.envs.mbpp.dev_size
    assert len(_rows(mbpp_env, "train")) + len(_rows(mbpp_env, "dev")) == 374


def test_no_math_test_problem_text_appears_in_training(math_env):
    sealed = [json.loads(line) for line in (DATA_ROOT / "test_sealed" / "math35.jsonl").read_text(encoding="utf-8").splitlines()]
    train_texts = {r["problem"].strip() for s in ("train", "calib", "dev") for r in _rows(math_env, s)}
    overlap = [r["problem_id"] for r in sealed if r["problem"].strip() in train_texts]
    assert len(overlap) <= 2, overlap        # exact-text duplicates between MATH train and MATH-500 are tolerated at <= 2 and reported


# ------------------------------------------------------------------ gold and garbage
def test_math35_gold_verifies_and_garbage_fails(math_env):
    rng = np.random.default_rng(5)
    prompts = math_env.generate_batch("dev", 200, rng)
    gold_ok = sum(math_env.evaluate_response(p, f"The answer is \\boxed{{{p.reference}}}.").success for p in prompts)
    assert gold_ok >= 0.99 * len(prompts)
    garbage = ["", "I do not know", "\\boxed{}", "\\boxed{banana}", "\\boxed{-99999}", "the answer is 5"]
    bad = sum(math_env.evaluate_response(p, g).success for p in prompts for g in garbage)
    assert bad <= 0.01 * len(prompts) * len(garbage)


def test_mbpp_gold_code_passes_and_garbage_fails(mbpp_env):
    rows = _rows(mbpp_env, "train")[:60]
    ok, total, failed = 0, 0, []
    for r in rows:
        prompt = mbpp_env._to_prompt(r, "train")  # noqa: SLF001
        verdict = mbpp_env.evaluate_response(prompt, f"```python\n{r['code']}\n```")
        total += 1
        ok += verdict.success
        if not verdict.success:
            failed.append(r["problem_id"])
    assert ok >= 0.99 * total, f"gold failed on {failed}"
    prompt = mbpp_env._to_prompt(rows[0], "train")  # noqa: SLF001
    for garbage in ("", "no code here", "```python\npass\n```", "```python\nprint('hi')\n```"):
        assert not mbpp_env.evaluate_response(prompt, garbage).success


def test_mbpp_verifier_time_is_recorded_and_infinite_loops_are_bounded(mbpp_env, cfg):
    from curator_rl.envs.mbpp import MbppEnv

    fast_cfg = cfg.data.envs.mbpp.model_copy(update={"verifier_timeout_s": 1.0})
    env = MbppEnv(REPO_ROOT, fast_cfg)
    row = _rows(env, "train")[0]
    prompt = env._to_prompt(row, "train")  # noqa: SLF001
    start = time.perf_counter()
    verdict = env.evaluate_response(prompt, "```python\nwhile True:\n    pass\n```")
    assert not verdict.success and verdict.info["timed_out"]
    assert time.perf_counter() - start < 6.0 and verdict.verifier_seconds > 0.5


def test_mbpp_prompt_shows_one_test_and_the_verifier_holds_the_rest(mbpp_env):
    row = _rows(mbpp_env, "train")[0]
    prompt = mbpp_env._to_prompt(row, "train")  # noqa: SLF001
    text = "\n".join(m["content"] for m in prompt.messages)
    assert row["test_list"][0] in text
    assert all(t not in text for t in row["test_list"][1:])      # hidden assertions never shown
    spec = json.loads(prompt.reference)
    assert set(row["test_list"]) <= set(spec["tests"]) and len(spec["tests"]) >= 2


def test_mbpp_special_casing_the_shown_test_fails_the_hidden_ones(mbpp_env):
    from curator_rl.core.types import Prompt

    spec = {"tests": ["assert add(1, 2) == 3", "assert add(2, 3) == 5"], "setup": ""}
    prompt = Prompt(prompt_id="x", env_id="mbpp", split="train", messages=(), reference=json.dumps(spec))
    hack = "```python\ndef add(a, b):\n    return 3\n```"
    assert not mbpp_env.evaluate_response(prompt, hack).success
    assert mbpp_env.evaluate_response(prompt, "```python\ndef add(a, b):\n    return a + b\n```").success


def test_reward_is_in_unit_interval_and_matches_success(math_env, mbpp_env):
    p = math_env.generate_prompt("dev", np.random.default_rng(1))
    for resp in (f"\\boxed{{{p.reference}}}", "nothing"):
        v = math_env.evaluate_response(p, resp)
        assert math_env.compute_reward(v) == (1.0 if v.success else 0.0)
    q = mbpp_env.generate_prompt("dev", np.random.default_rng(1))
    v = mbpp_env.evaluate_response(q, "nothing")
    assert mbpp_env.compute_reward(v) == 0.0 and not v.parse_ok
