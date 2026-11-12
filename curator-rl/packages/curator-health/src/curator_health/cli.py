"""curator-health CLI: ``preflight`` (probe verifiers) and ``audit`` (replay round moments)."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from curator_health.audit import PromptDependenceAudit, RoundMoments
from curator_health.config import HealthConfig
from curator_health.probe import ProbePrompt, probe_all


def _load_callable(path: str):
    module_name, _, attr = path.partition(":")
    if not attr:
        raise ValueError(f"expected 'module:attribute', got {path!r}")
    return getattr(importlib.import_module(module_name), attr)


def _load_spec(path: Path) -> tuple[dict, float]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    flag_rate = float(spec.get("flag_rate", 0.05))
    specs: dict[str, tuple] = {}
    for env_id, arm in spec["arms"].items():
        prompts = [ProbePrompt(prompt_id=p.get("prompt_id", ""), messages=tuple(p.get("messages", [])),
                               reference=str(p.get("reference", "")), meta=p.get("meta", {}))
                   for p in arm["prompts"]]
        specs[env_id] = (_load_callable(arm["verifier"]), prompts)
    return specs, flag_rate


def _cmd_preflight(args: argparse.Namespace) -> int:
    specs, flag_rate = _load_spec(args.spec)
    report = probe_all(specs, flag_rate=flag_rate)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["flagged"] else 0


def _cmd_audit(args: argparse.Namespace) -> int:
    cfg = HealthConfig(d_max=args.d_max, z=args.z, min_df=args.min_df)
    rows = [json.loads(line) for line in Path(args.observations).read_text(encoding="utf-8").splitlines() if line.strip()]
    env_ids = [r["env_id"] for r in rows]
    audit = PromptDependenceAudit(cfg, dict.fromkeys(env_ids), group_size=args.group_size)
    for r in rows:
        audit.observe(RoundMoments(**{k: r[k] for k in
                                      ("env_id", "n_prompts", "n_rollouts", "k_success", "sum_score", "sum_score_sq")}))
    out: dict[str, Any] = {"flagged": sorted(e for e, f in audit.flags().items() if f), "flags": audit.flags()}
    print(json.dumps(out, indent=2, sort_keys=True))
    return 1 if out["flagged"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="curator-health", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("preflight", help="probe each arm's reward path with junk answers")
    p_pre.add_argument("--spec", required=True, type=Path, help="JSON spec: arms -> {verifier, prompts}")
    p_pre.set_defaults(func=_cmd_preflight)

    p_aud = sub.add_parser("audit", help="replay round moments (JSONL) through the in-run audit")
    p_aud.add_argument("--observations", required=True, type=Path, help="JSONL of per-arm round moments")
    p_aud.add_argument("--group-size", type=int, default=8)
    p_aud.add_argument("--d-max", type=float, default=1.5)
    p_aud.add_argument("--z", type=float, default=2.33)
    p_aud.add_argument("--min-df", type=int, default=40)
    p_aud.set_defaults(func=_cmd_audit)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (KeyError, ValueError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
