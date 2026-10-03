"""Run metadata tests."""

import re

from conftest import BASE_CFG
from curator_rl.core import runmeta
from curator_rl.core.config import load_config

REQUIRED_KEYS = {
    "git_sha", "git_dirty", "diff_hash", "python", "platform",
    "packages", "gpu", "cpu_count", "timestamp_utc",
}


def test_run_metadata_required_keys(tmp_path):
    meta = runmeta.collect_run_metadata(tmp_path)
    assert REQUIRED_KEYS <= set(meta)
    assert isinstance(meta["git_dirty"], bool)
    assert meta["python"] and meta["platform"] and meta["packages"]


def test_run_id_format():
    cfg = load_config(BASE_CFG)
    meta = runmeta.collect_run_metadata()
    run_id = runmeta.make_run_id(cfg, meta, date="20261003")
    # {date}_{tier}_{method}_s{seed}_{git7}_{cfg6}
    assert re.fullmatch(r"20261003_\S+_curator_s0_(?:[0-9a-f]{7}|nogit)_[0-9a-f]{6}", run_id), run_id
