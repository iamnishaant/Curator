"""curator-health: label-free reward-health checks for RL with verifiable rewards (RLVR).

Two independent checks on one question: "does this reward mean anything?"

- ``probe_arm`` / ``probe_all`` (pre-flight): score junk answers through each arm's reward path.
- ``PromptDependenceAudit`` (in-run, free): flag a reward that shows no dependence on the prompt.

``RewardHealth`` is the facade that ties them together and is safe to adopt in ``shadow`` mode.
"""

from curator_health.audit import AuditResult, PromptDependenceAudit, RoundMoments
from curator_health.config import HealthConfig
from curator_health.health import RewardHealth
from curator_health.probe import ProbePrompt, probe_all, probe_arm, probes

HealthObservation = AuditResult   # the roadmap's name for one round's audit outcome

__version__ = "0.1.0"

__all__ = [
    "AuditResult",
    "HealthConfig",
    "HealthObservation",
    "ProbePrompt",
    "PromptDependenceAudit",
    "RewardHealth",
    "RoundMoments",
    "probe_all",
    "probe_arm",
    "probes",
    "__version__",
]
