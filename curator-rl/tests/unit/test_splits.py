"""Unit tests for split assignment and manifests (Roadmap D.3, Phase 2)."""

import pytest

from curator_rl.envs.splits import (
    assert_disjoint,
    assign_splits_finite,
    read_manifest_ids,
    split_hash_u,
    verify_manifests,
    write_manifests,
)


def _ids(n: int, prefix: str = "p") -> list[str]:
    return [f"{prefix}-{i}" for i in range(n)]


class TestSplitHash:
    def test_in_unit_interval(self):
        for i in range(200):
            u = split_hash_u("gsm8k", f"gsm8k-train-{i}", "salt")
            assert 0.0 <= u < 1.0

    def test_deterministic(self):
        assert split_hash_u("gsm8k", "x", "salt") == split_hash_u("gsm8k", "x", "salt")

    def test_salt_and_dataset_sensitive(self):
        assert split_hash_u("gsm8k", "x", "salt1") != split_hash_u("gsm8k", "x", "salt2")
        assert split_hash_u("gsm8k", "x", "salt") != split_hash_u("math", "x", "salt")
        assert split_hash_u("gsm8k", "x", "salt") != split_hash_u("gsm8k", "y", "salt")

    def test_matches_spec_formula(self):
        import hashlib

        digest = hashlib.sha256(b"saltgsm8kx").hexdigest()
        expected = int(digest[:8], 16) / 2**32
        assert split_hash_u("gsm8k", "x", "salt") == expected


class TestAssignSplits:
    def test_exact_sizes(self):
        ids = _ids(1000)
        out = assign_splits_finite("gsm8k", ids, "salt", 100, 150)
        counts = {s: sum(1 for v in out.values() if v == s) for s in ("train", "calib", "dev")}
        assert counts == {"train": 750, "calib": 100, "dev": 150}
        assert set(out) == set(ids)

    def test_deterministic_and_salt_sensitive(self):
        ids = _ids(500)
        a = assign_splits_finite("gsm8k", ids, "salt", 50, 50)
        b = assign_splits_finite("gsm8k", ids, "salt", 50, 50)
        c = assign_splits_finite("gsm8k", ids, "other-salt", 50, 50)
        assert a == b
        assert a != c

    def test_pool_too_small_raises(self):
        with pytest.raises(ValueError):
            assign_splits_finite("gsm8k", _ids(10), "salt", 8, 8)

    def test_disjoint(self):
        ids = _ids(400)
        out = assign_splits_finite("gsm8k", ids, "salt", 40, 40)
        by_split: dict[str, list[str]] = {}
        for pid, split in out.items():
            by_split.setdefault(split, []).append(pid)
        assert_disjoint(by_split)


class TestManifests:
    def test_write_read_verify_roundtrip(self, tmp_path):
        hashes = write_manifests(tmp_path, "gsm8k", {"train": ["b", "a"], "calib": ["c"]})
        assert set(hashes) == {"gsm8k_train.txt", "gsm8k_calib.txt"}
        assert read_manifest_ids(tmp_path, "gsm8k", "train") == ["a", "b"]  # sorted

        # no index yet -> verify reports it missing
        assert verify_manifests(tmp_path) == ["MANIFEST.sha256 missing"]

        from curator_rl.envs.splits import rewrite_manifest_index

        rewrite_manifest_index(tmp_path, hashes)
        assert verify_manifests(tmp_path) == []

    def test_verify_detects_tampering(self, tmp_path):
        from curator_rl.envs.splits import rewrite_manifest_index

        hashes = write_manifests(tmp_path, "gsm8k", {"train": ["a"]})
        rewrite_manifest_index(tmp_path, hashes)
        (tmp_path / "manifests" / "gsm8k_train.txt").write_text("a\nb\n", encoding="utf-8")
        assert verify_manifests(tmp_path) == ["manifest hash mismatch: gsm8k_train.txt"]

    def test_missing_manifest_file_detected(self, tmp_path):
        from curator_rl.envs.splits import rewrite_manifest_index

        hashes = write_manifests(tmp_path, "gsm8k", {"train": ["a"]})
        rewrite_manifest_index(tmp_path, hashes)
        (tmp_path / "manifests" / "gsm8k_train.txt").unlink()
        assert verify_manifests(tmp_path) == ["manifest file missing: gsm8k_train.txt"]


class TestBuiltSplitsIfPresent:
    """When the real splits have been built, verify the committed manifests."""

    def test_real_manifests_verify_and_are_disjoint(self):
        from pathlib import Path

        data_root = Path(__file__).resolve().parents[2] / "data"
        if not (data_root / "manifests" / "MANIFEST.sha256").exists():
            pytest.skip("splits not built yet (run scripts/build_splits.py)")
        assert verify_manifests(data_root) == []
        splits = {
            s: read_manifest_ids(data_root, "gsm8k", s) for s in ("train", "calib", "dev", "test")
        }
        assert_disjoint(splits)
        assert len(splits["test"]) == 1319  # official GSM8K test size
        assert len(splits["calib"]) == 300 and len(splits["dev"]) == 300
