"""Command-line interface: init-run, show-config, env-info."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from curator_rl.core import atomic, config, runmeta
from curator_rl.core import paths as paths_mod


def init_run(config_path: Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Create (or idempotently reuse) a run directory for the given config.

    Returns {"run_id", "run_dir", "config_hash"}. Raises CuratorConfigError if
    the directory exists with a different config.
    """
    overrides = overrides or []
    cfg = config.load_config(Path(config_path), overrides)
    cfg_hash = config.config_hash(cfg)

    runs_root = Path(cfg.paths.runs_root)
    run_id = runmeta.make_run_id(cfg, runmeta.collect_run_metadata())
    nd = paths_mod.RunPaths(runs_root, run_id)

    if nd.run_dir.exists():
        frozen_cfg_path = nd.run_dir / "config.yaml"
        if not frozen_cfg_path.exists():
            raise config.CuratorConfigError(
                f"run directory {nd.run_dir} exists but has no frozen config.yaml; remove it or pick another seed/name"
            )
        try:
            frozen = config.load_config(frozen_cfg_path)
        except config.CuratorConfigError:
            raise
        if config.config_hash(frozen) != cfg_hash:
            raise config.CuratorConfigError(
                f"config conflict: run {run_id} exists with a different config "
                f"(frozen hash {config.config_hash(frozen)[:6]}, requested {cfg_hash[:6]})"
            )
        return {"run_id": run_id, "run_dir": str(nd.run_dir), "config_hash": cfg_hash, "reused": True}

    nd.create()
    config.dump_config(cfg, nd.run_dir / "config.yaml")
    meta = runmeta.collect_run_metadata()
    meta["config_hash"] = cfg_hash
    atomic.atomic_write_json(nd.run_dir / "metadata.json", meta)
    return {"run_id": run_id, "run_dir": str(nd.run_dir), "config_hash": cfg_hash, "reused": False}


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="path to the YAML config file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. scheduler.tau=0.5")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="curator", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_init = subparsers.add_parser("init-run", help="create a run directory with a frozen config")
    _add_config_args(p_init)

    p_show = subparsers.add_parser("show-config", help="print the resolved, validated config as YAML")
    _add_config_args(p_show)

    subparsers.add_parser("env-info", help="print platform/git/hardware metadata as JSON")

    args = parser.parse_args(argv)

    try:
        if args.command == "init-run":
            result = init_run(Path(args.config), args.set)
            print(json.dumps(result))
            return 0
        if args.command == "show-config":
            import yaml

            cfg = config.load_config(Path(args.config), args.set)
            print(yaml.safe_dump(cfg.model_dump(by_alias=True, mode="json"), sort_keys=False))
            return 0
        if args.command == "env-info":
            print(json.dumps(runmeta.collect_run_metadata(), indent=2))
            return 0
    except config.CuratorConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
