"""GRPO group advantage statistics for binary rewards (Roadmap v3 §4.4, D-68).

SEC-style and DUMP-style allocators use the mean absolute advantage of an
environment's prompt groups as their learnability signal. For a group of G
binary rewards with k successes, GRPO's group-normalised advantage is
`(r - mean) / (std + eps)`; with the unbiased std used by TRL's default this
gives the closed form below, so the signal is computable from the pass counts
alone -- no trainer internals are needed.
"""

from __future__ import annotations

import math

GRPO_STD_EPS = 1e-4  # TRL's group-std epsilon


def group_mean_abs_advantage(k: int, g: int, *, eps: float = GRPO_STD_EPS) -> float:
    """Mean |advantage| over one group of `g` binary rewards with `k` successes.

    All-fail and all-pass groups have zero variance and therefore zero
    advantage. Otherwise, with p = k/g and the unbiased group std,
    mean|A| = 2 p (1 - p) / (std + eps).
    """
    if g < 2:
        raise ValueError("group size must be at least 2")
    if not 0 <= k <= g:
        raise ValueError("k must be in [0, g]")
    if k == 0 or k == g:
        return 0.0
    p = k / g
    std = math.sqrt(g / (g - 1) * p * (1.0 - p))
    return 2.0 * p * (1.0 - p) / (std + eps)
