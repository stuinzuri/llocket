"""Which engine versions this host runs: Ollama's (GET /api/version) and,
when the backends file lists a llama-server binary, that build's
(`llama-server --version`). Used for the engine-version run tag and the
API's GET /engines, so callers can split results on engine upgrades."""

from __future__ import annotations  # this account runs Python 3.9

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from typing import Optional

OLLAMA_TTL_S = 300          # an Ollama upgrade restarts it; re-ask every few minutes
_ollama_cache: dict = {}    # host -> (checked_at, version or None)
_binary_cache: dict = {}    # (path, mtime) -> version or None


def ollama_version(host: str, timeout: float = 5) -> Optional[str]:
    """e.g. "0.34.1", or None when Ollama doesn't answer."""
    hit = _ollama_cache.get(host)
    if hit and time.time() - hit[0] < OLLAMA_TTL_S:
        return hit[1]
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/version", timeout=timeout) as r:
            version = json.loads(r.read()).get("version") or None
    except (urllib.error.URLError, OSError, ValueError):
        version = None
    _ollama_cache[host] = (time.time(), version)
    return version


def llama_server_version(binary: str) -> Optional[str]:
    """The build of a llama-server binary, e.g. "b12284" (llama.cpp's own
    release names), from `--version`; None when it can't tell. Cached
    until the binary changes."""
    try:
        key = (binary, os.stat(binary).st_mtime)
    except OSError:
        return None
    if key in _binary_cache:
        return _binary_cache[key]
    try:
        p = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=20)
        out = p.stdout + p.stderr     # "version: 4321 (abc)" or "version: 0.9.0-dev (build 4321, commit abc)"
        m = re.search(r"\(build (\d+)", out) or re.search(r"version:\s*(\d+)\s*\(", out)
        version = f"b{m.group(1)}" if m else None
    except (OSError, subprocess.SubprocessError):
        version = None
    _binary_cache[key] = version
    return version


def engine_version(backend: str, host: str, binary: Optional[str] = None) -> Optional[str]:
    """The run tag value for a job's engine: "ollama-0.34.1" or
    "llama-server-b12284"; None when the version is unknown."""
    v = llama_server_version(binary) if backend == "llama-server" and binary else \
        ollama_version(host) if backend == "ollama" else None
    return f"{backend}-{v}" if v else None


def versions(host: str, binary: Optional[str] = None) -> dict:
    """{"ollama": {"host", "version"}, "llama_server": {"binary", "version"} or None}."""
    return {"ollama": {"host": host, "version": ollama_version(host)},
            "llama_server": {"binary": binary, "version": llama_server_version(binary)} if binary else None}
