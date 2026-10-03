"""Run directory layout: runs/{run_id}/{config.yaml, metadata.json, logs, checkpoints, reports}."""

from __future__ import annotations

from pathlib import Path

SUBDIRS = ("logs", "checkpoints", "reports")


class RunPaths:
    def __init__(self, runs_root: Path, run_id: str) -> None:
        self.runs_root = Path(runs_root)
        self.run_id = run_id
        self.run_dir = self.runs_root / run_id
        self.logs_dir = self.run_dir / "logs"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.reports_dir = self.run_dir / "reports"

    def create(self) -> None:
        """Idempotently create the run directory and its standard subdirectories."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in SUBDIRS:
            (self.run_dir / name).mkdir(exist_ok=True)
