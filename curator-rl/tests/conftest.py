from pathlib import Path

import pytest
import yaml

from curator_rl.core import config as config_mod

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_CFG = REPO_ROOT / "configs" / "base.yaml"
SMOKE_CFG = REPO_ROOT / "configs" / "experiment" / "smoke.yaml"


@pytest.fixture
def base_cfg() -> config_mod.RootConfig:
    return config_mod.load_config(BASE_CFG)


@pytest.fixture
def base_payload() -> dict:
    return yaml.safe_load(BASE_CFG.read_text(encoding="utf-8"))


def make_config_file(tmp_path: Path, overrides: dict | None = None, name: str = "cfg.yaml") -> Path:
    payload = yaml.safe_load(BASE_CFG.read_text(encoding="utf-8"))
    if overrides:
        for dotted, value in overrides.items():
            node = payload
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def assert_conflict_raises(config_path: Path, overrides: list[str], exc_cls: type = ValueError):
    with pytest.raises(exc_cls):
        config_mod.load_config(config_path, overrides)


def make_env_obs(env_id, *, n_prompts=4, group_size=8, success_rate=0.5, cost_usd=0.004):
    """Synthetic EnvRoundObs builder shared by the engine unit/integration tests."""
    from curator_rl.core.types import EnvRoundObs

    k = round(success_rate * n_prompts * group_size)
    mean_rate = success_rate
    n_rollouts = n_prompts * group_size
    mixed = n_prompts if 0.0 < success_rate < 1.0 else 0
    return EnvRoundObs(
        env_id=env_id,
        n_prompts=n_prompts,
        n_rollouts=n_rollouts,
        k_success=k,
        n_groups_mixed=mixed,
        sum_score=mean_rate * n_prompts,
        sum_score_sq=mean_rate * mean_rate * n_prompts,
        prompt_tokens=n_prompts * 256,
        completion_tokens=n_rollouts * 128,
        verifier_seconds=1e-4 * n_rollouts,
        gpu_seconds=cost_usd * 3600.0,
        cost_usd=cost_usd,
    )


def make_round_obs(envs, *, round_t, prompts_per_env=4, group_size=8, success_rate=0.5, cost_usd=0.004):
    """Full RoundObservation over `envs` (sorted ids), shared helper for tests."""
    from curator_rl.core.types import RoundObservation

    per_env = {
        env_id: make_env_obs(
            env_id, n_prompts=prompts_per_env, group_size=group_size,
            success_rate=success_rate, cost_usd=cost_usd,
        )
        for env_id in sorted(envs)
    }
    return RoundObservation(
        round=round_t,
        steps=round_t * 5,
        per_env=per_env,
        weights_used={env_id: 1.0 / len(envs) for env_id in sorted(envs)},
        round_cost_usd=cost_usd * len(envs),
        overhead_usd=0.0,
        budget_remaining_usd=10.0 - cost_usd * len(envs) * round_t,
    )


@pytest.fixture
def env_obs_factory():
    return make_env_obs


@pytest.fixture
def round_obs_factory():
    return make_round_obs


__all__ = [
    "REPO_ROOT", "BASE_CFG", "SMOKE_CFG", "base_cfg", "base_payload", "make_config_file",
    "assert_conflict_raises", "make_env_obs", "make_round_obs", "env_obs_factory", "round_obs_factory",
    "config_mod",
]
