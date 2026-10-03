"""JSONL writer/reader tests."""

import pytest

from curator_rl.core.jsonl import JsonlWriter, read_jsonl


def test_jsonl_roundtrip_and_append(tmp_path):
    path = tmp_path / "rounds.jsonl"
    with JsonlWriter(path, schema_version=1) as w:
        w.write({"round": 0, "weights": [0.5, 0.5]})
        w.write({"round": 1, "weights": [0.4, 0.6]})
    records = list(read_jsonl(path))
    assert [r["round"] for r in records] == [0, 1]
    assert [r["_seq"] for r in records] == [1, 2]
    assert records[0]["_schema"] == 1

    with JsonlWriter(path, schema_version=1) as w:
        w.write({"round": 2, "weights": [0.3, 0.7]})
    records = list(read_jsonl(path))
    assert [r["_seq"] for r in records] == [1, 2, 3]
    assert records[-1]["round"] == 2


def test_jsonl_truncated_tail_skipped(tmp_path):
    path = tmp_path / "cut.jsonl"
    with JsonlWriter(path, schema_version=2) as w:
        w.write({"a": 1})
        w.write({"a": 2})
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"a": 3, "half')  # simulates a crash mid-write

    skips: list[int] = []
    records = list(read_jsonl(path, on_skip=skips.append))
    assert len(records) == 2
    assert records[-1]["a"] == 2
    assert skips == [1]

    with pytest.raises(Exception):  # noqa: B017 - strict tail refusal raises json.JSONDecodeError
        list(read_jsonl(path, skip_corrupt_tail=False))
