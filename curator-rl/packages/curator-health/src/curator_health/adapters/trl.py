"""TRL / GRPO integration: wrap a reward function, feed the in-run audit, log the flags.

TRL calls the reward function once per generation batch with ``(prompts, completions, **dataset_columns)``
and uses the returned list as the reward. This adapter calls the user's function and returns **the exact
same object** it produced -- the value your trainer sees is byte-for-byte what it would be without us --
then folds the batch's per-arm statistics into a :class:`~curator_health.health.RewardHealth`.

Usage (one line changes):

    from curator_health import RewardHealth
    from curator_health.adapters.trl import RewardHealthCallback, wrap_trl_reward

    health = RewardHealth(env_ids=[], group_size=8)          # arms self-register on first sight
    trainer = GRPOTrainer(..., reward_funcs=wrap_trl_reward(my_reward_fn, health, group_size=8),
                          callbacks=[RewardHealthCallback(health)])

``env_id`` and ``prompt_id`` dataset columns are used when present; without them the adapter assumes a
single arm and groups completions in blocks of ``group_size`` (GRPO keeps a prompt's completions
contiguous), so it works on an unmodified dataset. Extra kwargs from TRL (e.g. ``completion_ids``) are
ignored.

Only the callback needs ``transformers``; it degrades to a compatible shim when it is absent, so this
module imports in a bare environment.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

from curator_health.health import RewardHealth

try:  # optional: the wrapper needs nothing, the callback subclasses this when available
    from transformers import TrainerCallback as _TrainerCallback
except Exception:  # noqa: BLE001 - keep the package importable without transformers
    class _TrainerCallback:  # minimal duck-typed stand-in
        def __init__(self) -> None:
            self.state = None
            self.control = None

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001, ANN201
            return control


def _column(columns: Mapping[str, Any], name: str | None, n: int) -> list[Any] | None:
    if name is None:
        return None
    col = columns.get(name)
    if col is None:
        return None
    col = list(col)
    return col if len(col) == n else None


def _spans(prompt_ids: Sequence[Any] | None, n: int, group_size: int) -> Iterator[tuple[int, int]]:
    """Contiguous prompt groups; without prompt ids, fixed blocks of group_size (GRPO order)."""
    g = max(int(group_size), 2)
    if prompt_ids is None:
        for i in range(0, n, g):
            yield i, min(i + g, n)
        return
    i = 0
    while i < n:
        j = i + 1
        while j < n and prompt_ids[j] == prompt_ids[i] and j - i < g:
            j += 1
        yield i, j
        i = j


class RewardHealthWrapper:
    """Callable with TRL's reward-function signature. Returns the host's rewards unchanged."""

    __name__ = "curator_health_reward"   # TRL names reward functions in its logs

    def __init__(self, reward_fn: Callable[..., list], health: RewardHealth, *,
                 group_size: int | None = None, env_id_column: str | None = "env_id",
                 prompt_id_column: str | None = "prompt_id",
                 default_env: str = "default") -> None:
        self.reward_fn = reward_fn
        self.health = health
        # default to the audit's G so blocks match the variance model
        self.group_size = int(group_size or health.audit.group_size)
        self.env_id_column = env_id_column
        self.prompt_id_column = prompt_id_column
        self.default_env = default_env
        self._announced: set[str] = set()

    def __call__(self, prompts=None, completions=None, **columns) -> list:
        rewards = self.reward_fn(prompts, completions, **columns)   # host exceptions propagate unchanged
        try:
            self._record(list(rewards or []), columns)
            self._announce()
        except Exception:  # noqa: BLE001 - the health check must never affect the reward path
            self.health.errors += 1
        return rewards

    def _record(self, rewards: list, columns: Mapping[str, Any]) -> None:
        n = len(rewards)
        if n == 0:
            return
        env_col = _column(columns, self.env_id_column, n)
        pid_col = _column(columns, self.prompt_id_column, n)
        stats: dict[str, list[float]] = {}
        for start, end in _spans(pid_col, n, self.group_size):
            g = end - start
            if g < 2:
                continue
            k = sum(1 for r in rewards[start:end] if (r or 0.0) >= 0.5)
            rate = k / g
            env = str(env_col[start]) if env_col is not None else self.default_env
            s = stats.setdefault(env, [0, 0, 0, 0.0, 0.0])
            s[0] += 1
            s[1] += g
            s[2] += k
            s[3] += rate
            s[4] += rate * rate
        for env, (n_prompts, n_rollouts, k_success, sum_score, sum_score_sq) in stats.items():
            self.health.observe(env_id=env, n_prompts=int(n_prompts), n_rollouts=int(n_rollouts),
                                k_success=int(k_success), sum_score=sum_score, sum_score_sq=sum_score_sq)

    def _announce(self) -> None:
        if self.health.mode == "shadow":
            return
        for env in self.health.flagged():
            if env not in self._announced:
                self._announced.add(env)
                print(f"[curator-health] reward for '{env}' shows no dependence on the prompt "
                      f"(mode={self.health.mode})", file=sys.stderr)


def wrap_trl_reward(reward_fn: Callable[..., list], health: RewardHealth, *,
                    group_size: int | None = None, env_id_column: str | None = "env_id",
                    prompt_id_column: str | None = "prompt_id") -> RewardHealthWrapper:
    """Wrap a TRL GRPO reward function so it also feeds the in-run reward audit."""
    return RewardHealthWrapper(reward_fn, health, group_size=group_size, env_id_column=env_id_column,
                               prompt_id_column=prompt_id_column)


def health_metrics(health: RewardHealth) -> dict[str, float]:
    """Scalars for the trainer's log history: how many arms are flagged and which."""
    flags = health.flags()
    out: dict[str, float] = {
        "health/flagged": float(sum(1 for f in flags.values() if f)),
        "health/arms": float(len(flags)),
        "health/errors": float(health.errors),
    }
    out.update({f"health/flag/{env}": float(f) for env, f in flags.items()})
    return out


class RewardHealthCallback(_TrainerCallback):
    """Logs the health verdict into every TRL ``log_history`` row (and warns in warn/gate mode)."""

    def __init__(self, health: RewardHealth) -> None:
        super().__init__()
        self.health = health

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001, ANN201
        if logs is not None:
            logs.update(health_metrics(self.health))
        if self.health.mode in ("warn", "gate"):
            for env in self.health.flagged():
                print(f"[curator-health] flagged arm: {env}", file=sys.stderr)
        return control
