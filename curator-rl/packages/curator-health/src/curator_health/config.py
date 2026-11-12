"""Configuration for the reward-health checks.

Defaults are the ones validated in the CURATOR project (DECISIONS.md D-98, D-102): ``d_max = 1.5``
separates a prompt-independent reward (D = 1) from a real verifier (D = 3.0-4.4), and ``flag_rate``
is the junk-reward share above which the pre-flight probe flags an arm.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HealthConfig:
    """Thresholds for both checks. Immutable; validation happens on construction."""

    d_max: float = 1.5          # dispersion ratio separating prompt-independent (1) from real (3-4) rewards
    z: float = 2.33             # one-sided confidence multiplier on log D (enter below, clear above)
    min_df: int = 40            # pooled degrees of freedom before any in-run decision
    flag_rate: float = 0.05     # junk-reward rate above which the pre-flight probe flags an arm
    preflight_gate: bool = False  # start a probe-flagged arm flagged for the whole run

    def __post_init__(self) -> None:
        if not self.d_max > 1.0:
            raise ValueError(f"d_max must be > 1, got {self.d_max}")
        if not self.z > 0.0:
            raise ValueError(f"z must be > 0, got {self.z}")
        if self.min_df < 2:
            raise ValueError(f"min_df must be >= 2, got {self.min_df}")
        if not 0.0 <= self.flag_rate <= 1.0:
            raise ValueError(f"flag_rate must be in [0, 1], got {self.flag_rate}")
