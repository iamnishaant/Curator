"""Atomic write tests."""


from curator_rl.core.atomic import atomic_write_json, atomic_write_text


def test_atomic_write_no_partial_file(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    atomic_write_text(path, "v1")
    assert path.read_text(encoding="utf-8") == "v1"

    # simulate a crash between temp-file creation and rename
    import curator_rl.core.atomic as atomic_mod

    orig_replace = atomic_mod.os.replace

    def boom(*a, **k):
        raise OSError("simulated crash")

    monkeypatch.setattr(atomic_mod.os, "replace", boom)
    try:
        atomic_write_text(path, "v2-full-but-never-renamed")
    except OSError:
        pass
    finally:
        monkeypatch.setattr(atomic_mod.os, "replace", orig_replace)

    # old content intact, no temp file left behind
    assert path.read_text(encoding="utf-8") == "v1"
    leftovers = [p.name for p in path.parent.iterdir() if ".tmp" in p.name]
    assert leftovers == [], leftovers


def test_atomic_write_json_roundtrip(tmp_path):
    path = tmp_path / "meta.json"
    atomic_write_json(path, {"b": 1, "a": {"nested": [1, 2, 3]}})
    import json

    assert json.loads(path.read_text(encoding="utf-8"))["a"]["nested"] == [1, 2, 3]
