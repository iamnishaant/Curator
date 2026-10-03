"""Append-only JSONL logging with a schema version and a tolerant reader."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any


class JsonlWriter:
    """Append-only JSONL writer. Records get `_schema` and a monotonic `_seq`.

    Reopening an existing file appends and continues `_seq`, so a crash never
    reuses a sequence number.
    """

    def __init__(self, path: Path, schema_version: int, fsync: bool = False) -> None:
        self.path = Path(path)
        self.schema_version = int(schema_version)
        self.fsync = bool(fsync)
        self._seq = 0
        self._fh: Any = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.__enter__()

    def _next_seq(self) -> int:
        highest = 0
        if self.path.exists():
            for record in read_jsonl(self.path):
                seq = record.get("_seq")
                if isinstance(seq, int) and seq > highest:
                    highest = seq
        return highest

    def write(self, record: dict) -> None:
        self._seq += 1
        payload = {"_schema": self.schema_version, "_seq": self._seq, **record}
        self._fh.write(json.dumps(payload, sort_keys=False, default=str) + "\n")
        self._fh.flush()
        if self.fsync:
            os.fsync(self._fh.fileno())

    def __enter__(self) -> JsonlWriter:
        if self._fh is None:
            self._seq = self._next_seq()
            self._fh = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc: object) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def read_jsonl(
    path: Path,
    skip_corrupt_tail: bool = True,
    on_skip: Callable[[int], None] | None = None,
) -> Iterator[dict]:
    """Yield records; a truncated/corrupt tail is skipped (and reported) or raises."""
    skipped = 0
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if not skip_corrupt_tail:
                    raise
                skipped += 1
                break
    if skipped and on_skip is not None:
        on_skip(skipped)
