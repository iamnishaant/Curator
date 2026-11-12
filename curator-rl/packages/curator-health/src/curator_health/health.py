"""``RewardHealth``: the integration facade over the two checks.

Modes (so the package can be adopted without a leap of faith):

- ``shadow`` (default): observe, flag and report; offers the host no way to act on a flag.
- ``warn``:  as shadow, but ``flags()``/``report()`` are meant to be surfaced loudly.
- ``gate``:  as warn; the host is expected to read ``flags()`` and act on it (drop an arm, alert, ...).

The package never mutates a reward, a weight or a trainer. Every entry point is fail-open: an internal
error is counted and swallowed, never raised into a training run.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from curator_health.audit import AuditResult, PromptDependenceAudit, RoundMoments
from curator_health.config import HealthConfig
from curator_health.probe import ProbePrompt, RewardFn, probe_all

_FIELDS = ("env_id", "n_prompts", "n_rollouts", "k_success", "sum_score", "sum_score_sq")
_MODES = ("shadow", "warn", "gate")


class RewardHealth:
    """Stateful health monitor for one run: pre-flight (once) plus the in-run audit (per round)."""

    def __init__(self, *, env_ids: Iterable[str], group_size: int,
                 config: HealthConfig | None = None, mode: str = "shadow") -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.config = config or HealthConfig()
        self.mode = mode
        self.audit = PromptDependenceAudit(self.config, env_ids, group_size)
        self.errors = 0
        self.preflight_report: dict[str, Any] | None = None

    @property
    def acts(self) -> bool:
        return self.mode == "gate"

    def preflight(self, specs: Mapping[str, tuple[RewardFn, list[ProbePrompt]]]) -> dict[str, Any]:
        """Probe each arm's reward path with junk answers. Optional; run once, before training."""
        report = probe_all(specs, flag_rate=self.config.flag_rate)
        if self.config.preflight_gate:
            self.audit.preflag(report["flagged"])
        self.preflight_report = report
        return report

    def observe(self, obs: RoundMoments | None = None, **moments: Any) -> AuditResult:
        """Feed one arm's round statistics. Never raises; on error returns a harmless result.

        Accepts either a ``RoundMoments`` or the six fields as keyword arguments; unknown keywords
        (such as ``round``) are ignored.
        """
        if obs is None:
            try:
                obs = RoundMoments(**{k: moments[k] for k in _FIELDS})
            except (KeyError, TypeError):
                self.errors += 1
                return AuditResult(float("nan"), 0, False)
        try:
            return self.audit.observe(obs)
        except Exception:  # noqa: BLE001 - fail-open
            self.errors += 1
            return AuditResult(float("nan"), 0, False)

    def flags(self) -> dict[str, bool]:
        return self.audit.flags()

    def flagged(self) -> list[str]:
        return sorted(e for e, f in self.flags().items() if f)

    def report(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "acts": self.acts,
            "flagged": self.flagged(),
            "flags": self.flags(),
            "errors": self.errors,
            "preflight": self.preflight_report,
        }

    def get_state(self) -> dict[str, Any]:
        return {"audit": self.audit.get_state(), "errors": self.errors}

    def load_checkpoint(self, state: Mapping[str, Any]) -> None:
        self.audit.load_checkpoint(state["audit"])
        self.errors = int(state.get("errors", 0))
