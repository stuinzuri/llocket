"""Shared primitives for Llocket's filesystem-as-queue mechanism.

Every individual file write goes through write-to-temp-then-rename (never
an in-place overwrite); every job-state transition is one whole-directory
rename (never separate renames of sibling files). See README.md for the
design rationale (NFS rename atomicity, sync-mode NFS export, why SQLite
isn't the source of truth here).
"""

import json
import os
import time
import urllib.error
import urllib.request

# Both overridable via env var so worker.py/api.py can run fully offline
# against a local directory + local Ollama for development (e.g. away
# from the production host/queue) without touching the real queue. The
# defaults are the production paths -- nothing changes for a real
# deployment, which sets neither var.
QUEUE_ROOT = os.environ.get("LLOCKET_QUEUE_ROOT", os.path.expanduser("~/llocket/nas/queue"))
OLLAMA_HOST = os.environ.get("LLOCKET_OLLAMA_HOST", "http://localhost:11437")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_queue_dirs() -> None:
    for sub in (".incoming", "queued", "running", "done", "failed"):
        os.makedirs(os.path.join(QUEUE_ROOT, sub), exist_ok=True)


def atomic_write_json(dir_path: str, filename: str, data: dict) -> None:
    """Write JSON to `filename` in `dir_path` via temp-file-then-rename --
    a reader never sees a partial/torn file under the real name."""
    real_path = os.path.join(dir_path, filename)
    tmp_path = real_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp_path, real_path)  # atomic, same filesystem


def read_json(dir_path: str, filename: str) -> dict:
    with open(os.path.join(dir_path, filename)) as f:
        return json.load(f)


def call_ollama(
    host: str,
    model: str,
    messages: list,
    think: bool = False,
    options: dict = None,
    stall_timeout: float = 600.0,
) -> dict:
    """Streaming chat call -- NOT the non-streaming form this started as.
    A real job can run tens of minutes to over an hour; a single
    non-streaming urlopen(timeout=N) is a hard cap on *total* call
    duration, which would kill every real production job outright.
    Streaming with urllib's per-read socket timeout gives a "stall
    detector, re-armed per chunk" semantics instead -- generation can run
    indefinitely as long as *some* byte arrives at least every
    stall_timeout seconds.

    `think=False` by default -- leaving this unset lets a thinking-
    capable model burn its whole num_predict budget on reasoning tokens
    and emit content="" with done_reason="length" on a small test
    budget, which looks like a silent failure but isn't one. Production
    callers with a real (large) num_ctx/num_predict budget should pass
    think=True if they want native thinking captured separately
    (message.thinking) rather than disabled.

    `options` is passed through to Ollama verbatim (num_predict,
    temperature, top_p, repeat_penalty, num_ctx, whatever) -- stays a
    generic pass-through rather than named parameters so a job's
    input.json can carry any sampling_options dict without this function
    needing to know every possible key in advance.

    Returns a dict shaped like Ollama's own non-streaming response
    (aggregated message.content/message.thinking, done_reason, and the
    usual stats fields) so callers don't need to know streaming was used
    internally."""
    payload = {"model": model, "messages": messages, "stream": True, "think": think}
    if options:
        payload["options"] = options

    req = urllib.request.Request(
        f"{host}/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )

    content_parts = []
    thinking_parts = []
    final: dict = {}
    with urllib.request.urlopen(req, timeout=stall_timeout) as resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "error" in obj:
                raise OSError(f"Ollama stream error: {obj['error']}")
            message = obj.get("message", {}) or {}
            content_parts.append(message.get("content") or "")
            thinking_parts.append(message.get("thinking") or "")
            if obj.get("done"):
                final = obj

    return {
        "message": {"content": "".join(content_parts), "thinking": "".join(thinking_parts) or None},
        "done_reason": final.get("done_reason"),
        "prompt_eval_count": final.get("prompt_eval_count"),
        "eval_count": final.get("eval_count"),
        "total_duration": final.get("total_duration"),
        "load_duration": final.get("load_duration"),
        "prompt_eval_duration": final.get("prompt_eval_duration"),
        "eval_duration": final.get("eval_duration"),
    }
