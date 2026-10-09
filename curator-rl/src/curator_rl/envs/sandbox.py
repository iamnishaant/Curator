"""Python code sandbox for the MBPP environment (Roadmap v3 Phase E, D.2).

Runs model-written code plus assertions in a separate interpreter process with a wall-clock
timeout, a memory and CPU limit (POSIX) and soft guards against network access and spawning
processes. Success is decided by a RANDOM per-run sentinel printed only after every assertion
has executed, never by the exit code: code that calls `exit()` / `os._exit(0)` before the
tests produces no sentinel and fails; a program cannot print a sentinel it cannot guess.

THIS IS NOT A SECURITY BOUNDARY. The guards stop honest mistakes and cheap reward hacks (hanging,
spawning, network, early exit); a determined program can still read files or exhaust the
machine. Run it only on disposable machines (Kaggle sessions, containers), never on a
workstation with secrets. Cross-platform: the timeout and sentinel work on Windows; resource
limits and process-group kill need POSIX (GPU runs are on Linux, Roadmap O.1).
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

_PREAMBLE = '''\
import socket as _socket, subprocess as _subprocess, os as _os
def _blocked(*_a, **_k):
    raise OSError("disabled in the sandbox")
for _n in ("socket", "create_connection", "getaddrinfo", "gethostbyname"):
    setattr(_socket, _n, _blocked)
for _n in ("Popen", "run", "call", "check_call", "check_output"):
    setattr(_subprocess, _n, _blocked)
for _n in ("system", "popen", "fork", "forkpty", "execv", "execve", "execl", "execvp", "spawnl", "spawnv"):
    if hasattr(_os, _n):
        setattr(_os, _n, _blocked)
del _socket, _subprocess, _os, _n
'''


@dataclass(frozen=True)
class SandboxResult:
    passed: bool
    timed_out: bool
    seconds: float
    returncode: int | None
    stderr_tail: str


def _limits_preamble(timeout_s: float, memory_mb: int | None) -> str:
    """Resource limits applied by the child interpreter itself, before any model code runs.

    Set inside the child (not via `preexec_fn`) so `run_python_tests` is safe to call from a
    thread pool: `preexec_fn` is unsafe in a multi-threaded parent (it can deadlock after fork).
    """
    if os.name != "posix":
        return ""
    cpu = (int(timeout_s) + 1, int(timeout_s) + 2)
    fsize = 64 * 1024 * 1024
    lines = [
        "import resource as _resource",
        f"_resource.setrlimit(_resource.RLIMIT_CPU, {cpu!r})",
        f"_resource.setrlimit(_resource.RLIMIT_FSIZE, ({fsize}, {fsize}))",
    ]
    if memory_mb:
        cap = memory_mb * 1024 * 1024
        lines.append(f"_resource.setrlimit(_resource.RLIMIT_AS, ({cap}, {cap}))")
    lines.append("del _resource")
    return "\n".join(lines) + "\n"


def run_python_tests(
    code: str,
    tests: list[str],
    setup: str = "",
    *,
    timeout_s: float = 5.0,
    memory_mb: int | None = 512,
) -> SandboxResult:
    """Execute `code`, then `setup`, then each assertion in `tests`; pass iff the sentinel appears.

    Setup runs AFTER the solution (standard MBPP harness order): some tasks build test objects from a
    class the solution defines (e.g. mbpp-927). Thread-safe: many calls may run concurrently.
    """
    sentinel = "SANDBOX_OK_" + secrets.token_hex(12)
    preamble = _limits_preamble(timeout_s, memory_mb) + _PREAMBLE
    program = "\n".join([preamble, code, "", setup or "", "", *tests, f"print({sentinel!r})", ""])
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="curator_sbx_") as tmp:
        script = os.path.join(tmp, "prog.py")
        out_path = os.path.join(tmp, "out.txt")
        err_path = os.path.join(tmp, "err.txt")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(program)
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8",
               "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}
        try:
            with open(out_path, "wb") as out, open(err_path, "wb") as err:
                proc = subprocess.run(
                    [sys.executable, "-I", script], stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                    cwd=tmp, env=env, timeout=timeout_s, start_new_session=(os.name == "posix"),
                    check=False,
                )
            timed_out, rc = False, proc.returncode
        except subprocess.TimeoutExpired:
            timed_out, rc = True, None
        stdout = _tail(out_path)
        stderr = _tail(err_path)
    passed = (not timed_out) and rc == 0 and sentinel in stdout
    return SandboxResult(passed=passed, timed_out=timed_out, seconds=time.perf_counter() - start,
                         returncode=rc, stderr_tail=stderr[-400:])


def _tail(path: str, limit: int = 65536) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(size - limit, 0))
            return fh.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
