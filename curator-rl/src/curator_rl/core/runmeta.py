"""Run metadata: git state, platform, packages, hardware; plus run IDs."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

_TRACKED_PACKAGES = ("curator-rl", "numpy", "scipy", "pandas", "pyarrow", "pydantic", "pyyaml")


def _run_git(*args: str, cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            # git emits UTF-8; the locale codec (cp1252 on Windows) cannot decode
            # characters such as 'ρ' or '≥' in a dirty-tree diff and left stdout None
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or result.stdout is None:
        return None
    return result.stdout.strip()


def _gpu_info() -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        gpus.append({"name": parts[0] if parts else None, "memory_total": parts[1] if len(parts) > 1 else None})
    return {"list": gpus, "driver_version": parts[-1] if len(parts) > 2 else None}


def _installed_packages() -> dict[str, str]:
    out = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = pkg_version(name)
        except PackageNotFoundError:
            out[name] = "not-installed"
    return out


def collect_run_metadata(repo_root: Path | None = None) -> dict[str, Any]:
    repo_root = Path(repo_root) if repo_root is not None else Path.cwd()

    git_sha = _run_git("rev-parse", "HEAD", cwd=repo_root)
    if git_sha is not None:
        dirty_status = _run_git("status", "--porcelain", cwd=repo_root)
        git_dirty = dirty_status is not None and dirty_status != ""
        diff_output = _run_git("diff", cwd=repo_root) or ""
        diff_hash = hashlib.sha256(diff_output.encode("utf-8")).hexdigest()[:16]
    else:
        git_dirty = False
        diff_hash = None

    return {
        "git_sha": git_sha,
        "git_dirty": bool(git_dirty),
        "diff_hash": diff_hash,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _installed_packages(),
        "gpu": _gpu_info(),
        "cpu_count": __import__("os").cpu_count(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }


def make_run_id(cfg, meta: dict[str, Any], date: str | None = None) -> str:
    """{date}_{tier}_{method}_s{seed}_{git7}_{cfg6} — cwd/Git-free and deterministic."""
    from .config import config_hash  # relative import keeps core self-contained

    date_part = date if date is not None else datetime.now(UTC).strftime("%Y%m%d")
    tier_part = str(cfg.experiment.tier)
    git7 = (meta.get("git_sha") or "nogit")[:7]
    cfg6 = config_hash(cfg)[:6]
    return f"{date_part}_{tier_part}_{cfg.experiment.method}_s{cfg.experiment.seed}_{git7}_{cfg6}"
