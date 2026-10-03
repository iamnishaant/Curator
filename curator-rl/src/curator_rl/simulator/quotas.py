"""Weight-to-quota conversion (Roadmap D.4): largest-remainder rounding.

`m_i = round(M * w_i)` with the residual distributed by largest remainder and
a seeded tie-break, so `sum(m_i) == M` exactly and the mapping is
deterministic. `stochastic` rounding is left as a Phase 8 option.
"""

from __future__ import annotations

import numpy as np


def largest_remainder_quotas(
    weights: dict[str, float],
    total: int,
    *,
    tie_break_seed: int = 0,
) -> dict[str, int]:
    """Integer quotas summing exactly to `total` (Roadmap D.4)."""
    if total < 0:
        raise ValueError("total must be non-negative")
    if not weights:
        return {}
    keys = sorted(weights)
    exact = np.array([weights[k] * total for k in keys], dtype=float)
    floors = np.floor(exact).astype(int)
    remainder = total - int(floors.sum())
    if remainder > 0:
        fracs = exact - floors
        # deterministic tie-break: jitter by hash order, then stable sort
        rng = np.random.default_rng(abs(tie_break_seed) + 1)
        jitter = rng.uniform(0, 1e-9, size=len(keys))
        order = np.argsort(-(fracs + jitter), kind="stable")
        for idx in order[:remainder]:
            floors[idx] += 1
    return {k: int(v) for k, v in zip(keys, floors)}
