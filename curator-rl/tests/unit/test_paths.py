"""RunPaths tests."""

from pathlib import Path

from curator_rl.core.paths import RunPaths


def test_runpaths_create_idempotent(tmp_path):
    rp = RunPaths(Path(tmp_path), "20261003_t0_smoke_s0_ab12cd3_9f31e2")
    rp.create()
    first = sorted(p.name for p in rp.run_dir.iterdir())
    rp.create()  # second call is a no-op
    second = sorted(p.name for p in rp.run_dir.iterdir())
    assert first == second == ["checkpoints", "logs", "reports"]
    assert rp.logs_dir.exists() and rp.run_dir.exists()
