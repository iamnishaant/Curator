"""Unit tests for the Python sandbox and MBPP code extraction (Phase E, no data needed)."""

import os
import time

import pytest

from curator_rl.envs.mbpp import extract_code
from curator_rl.envs.sandbox import run_python_tests

ADD = "def add(a, b):\n    return a + b\n"
TESTS = ["assert add(1, 2) == 3", "assert add(2, 3) == 5"]


def test_correct_code_passes_and_wrong_code_fails():
    assert run_python_tests(ADD, TESTS).passed
    assert not run_python_tests("def add(a, b):\n    return a - b\n", TESTS).passed


def test_special_casing_the_shown_assertion_fails_the_hidden_ones():
    hack = "def add(a, b):\n    return 3\n"                    # passes assert add(1, 2) == 3 only
    assert run_python_tests(hack, TESTS[:1]).passed
    assert not run_python_tests(hack, TESTS).passed


def test_early_exit_cannot_score():
    for early in ("import sys\nsys.exit(0)\n", "import os\nos._exit(0)\n", "raise SystemExit(0)\n"):
        assert not run_python_tests(early + ADD, TESTS).passed


def test_guessed_sentinel_cannot_score():
    forged = "print('SANDBOX_OK_deadbeef')\ndef add(a, b):\n    return 0\n"
    assert not run_python_tests(forged, TESTS).passed


def test_infinite_loop_is_killed_within_the_timeout():
    start = time.perf_counter()
    res = run_python_tests("while True:\n    pass\n", ["assert True"], timeout_s=1.0)
    assert res.timed_out and not res.passed
    assert time.perf_counter() - start < 6.0


def test_network_and_process_spawning_are_blocked():
    net = "import socket\nsocket.create_connection(('example.com', 80), timeout=1)\n"
    spawn = "import subprocess\nsubprocess.run(['echo', 'x'])\n"
    for code in (net, spawn):
        res = run_python_tests(code, ["assert True"])
        assert not res.passed


def test_syntax_error_and_runtime_error_fail_cleanly():
    assert not run_python_tests("def add(:\n", TESTS).passed
    assert not run_python_tests("def add(a, b):\n    raise ValueError('x')\n", TESTS).passed


def test_setup_code_runs_before_the_solution():
    setup = "BASE = 10"
    code = "def f():\n    return BASE\n"
    assert run_python_tests(code, ["assert f() == 10"], setup).passed


def test_concurrent_calls_from_threads_are_independent():
    from concurrent.futures import ThreadPoolExecutor

    jobs = [(ADD, TESTS, True), ("def add(a, b):\n    return 0\n", TESTS, False)] * 4
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda j: run_python_tests(j[0], j[1]).passed, jobs))
    assert results == [j[2] for j in jobs]


@pytest.mark.skipif(os.name != "posix", reason="address-space limits are POSIX only")
def test_memory_hog_is_stopped():
    res = run_python_tests("x = bytearray(2 * 1024 ** 3)\n", ["assert True"], memory_mb=256)
    assert not res.passed


def test_code_extraction():
    assert extract_code("text\n```python\ndef f():\n    return 1\n```\nmore") == "def f():\n    return 1"
    assert extract_code("```python\na = 1\n```\n```python\ndef f():\n    pass\n```") == "def f():\n    pass"
    assert extract_code("def f():\n    return 2") == "def f():\n    return 2"
    assert extract_code("```python\ndef f():\n    return 3\n") == "def f():\n    return 3"   # unterminated fence
    assert extract_code("I think you should sort the list first.") is None
    assert extract_code("") is None
