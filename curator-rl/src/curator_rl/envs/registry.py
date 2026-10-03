"""Environment registry (Roadmap Phase 2).

Maps environment names to factory functions driven by the strictly
validated `data.envs.*` config sections. Environments are constructed
lazily so importing the registry stays cheap.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from curator_rl.envs.base import Environment

_FACTORY_SIG = Callable[[Path, object], Environment]
_FACTORIES: dict[str, _FACTORY_SIG] = {}


def _register(env_id: str, factory: _FACTORY_SIG) -> None:
    _FACTORIES[env_id] = factory


def _make_gsm8k(repo_root: Path, data_cfg) -> Environment:
    from curator_rl.envs.gsm8k import Gsm8kEnv

    return Gsm8kEnv(repo_root, data_cfg.envs.gsm8k)


def _make_noisy(repo_root: Path, data_cfg) -> Environment:
    from curator_rl.envs.noisy import NoisyRewardEnv

    return NoisyRewardEnv(repo_root, data_cfg.envs.noisy)


def _make_countdown(repo_root: Path, data_cfg) -> Environment:
    from curator_rl.envs.countdown import CountdownEnv

    return CountdownEnv(repo_root, data_cfg.envs.countdown)


_register("gsm8k", _make_gsm8k)
_register("noisy", _make_noisy)
_register("countdown", _make_countdown)


class EnvRegistry:
    def __init__(self, repo_root: Path, data_cfg) -> None:
        self._repo_root = Path(repo_root)
        self._data_cfg = data_cfg
        self._cache: dict[str, Environment] = {}

    def get(self, env_id: str) -> Environment:
        if env_id in self._cache:
            return self._cache[env_id]
        if env_id not in _FACTORIES:
            raise KeyError(f"unknown environment '{env_id}' (known: {sorted(_FACTORIES)})")
        env = _FACTORIES[env_id](self._repo_root, self._data_cfg)
        self._cache[env_id] = env
        return env

    def names(self) -> list[str]:
        return sorted(_FACTORIES)
