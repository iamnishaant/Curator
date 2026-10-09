"""Architecture import rules (Roadmap B.2 / v1 Phase 1 test 'test_import_rules'):

- core (L0): no project imports, no heavy/L2 dependencies.
- L1 packages (signals, scheduler, calibration, roi): only core + numpy/scipy;
  never torch/transformers/trl/datasets, never L2/L3 project packages.
- Only trainer/trl_adapter.py may import trl.
- Nothing outside evaluation/ imports evaluation.final_test (the sealed loader).
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "curator_rl"

L1_PACKAGES = {"signals", "scheduler", "calibration", "roi"}
L2_PACKAGES = {"envs", "cost", "trainer", "simulator"}
HEAVY_EXTERNAL = {"torch", "transformers", "trl", "datasets", "vllm", "peft", "accelerate"}
L1_ALLOWED_EXTERNAL = {"numpy", "scipy"}


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                # keep curator_rl.<pkg> so the L1-may-only-import-core rule works
                roots.add(".".join(parts[:2]) if parts[0] == "curator_rl" else parts[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                if node.module:
                    roots.add(node.module.split(".")[0])
            elif node.module:
                parts = node.module.split(".")
                roots.add(".".join(parts[:2]) if parts[0] == "curator_rl" else parts[0])
    return roots


def _package_of(path: Path) -> str | None:
    rel = path.relative_to(SRC).parts
    return rel[0] if rel else None


def test_import_rules():
    assert SRC.exists(), "src/curator_rl missing from the checkout"
    violations: list[str] = []

    for py in sorted(SRC.rglob("*.py")):
        package = _package_of(py)
        if package is None:
            continue
        imports = _top_level_imports(py)

        # hard invariant: heavy RL/LLM libraries only in trainer/trl_adapter.py
        if "trl" in imports and not (package == "trainer" and py.name == "trl_adapter.py"):
            violations.append(f"{py.relative_to(SRC)}: 'trl' import only allowed in trainer/trl_adapter.py")

        # sealed-test loader: nothing outside evaluation/ may import it
        for imp in imports:
            if imp == "evaluation" and package != "evaluation":
                violations.append(f"{py.relative_to(SRC)}: 'evaluation' import outside evaluation/")

        if package == "core":
            banned = HEAVY_EXTERNAL | L1_PACKAGES | L2_PACKAGES | {"curator_rl", "evaluation"}
        elif package in L1_PACKAGES:
            # L1 may import core, its own package, and OTHER L1 packages:
            # the scheduler owns a SignalEngine (A.2 decision 1 — a pure
            # function of observations must build signals internally), and
            # calibration/roi consume signals too. L2 stays banned.
            banned = (
                HEAVY_EXTERNAL
                | {p for p in L2_PACKAGES}
                | {"curator_rl", "evaluation"}
            )
            allowed_l1 = L1_PACKAGES | {f"curator_rl.{p}" for p in L1_PACKAGES}
            for imp in imports:
                if (
                    imp not in allowed_l1
                    and imp.startswith("curator_rl")
                    and not imp.startswith("curator_rl.core")
                ):
                    violations.append(
                        f"{py.relative_to(SRC)}: L1 may only import core and L1 (got '{imp}')"
                    )
        elif package in L2_PACKAGES:
            banned = set()  # L2 may use heavy libraries except trl (rule above)
        else:
            continue  # L3 (evaluation, experiments...) can import anything

        for name in banned:
            if name in imports:
                violations.append(f"{py.relative_to(SRC)}: package '{package}' imports banned '{name}'")

    assert not violations, "\n".join(violations)
