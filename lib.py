"""Shared primitives for Llocket's filesystem-as-queue mechanism.

Every individual file write goes through write-to-temp-then-rename (never
an in-place overwrite); every job-state transition is one whole-directory
rename (never separate renames of sibling files). See README.md for the
design rationale (NFS rename atomicity, sync-mode NFS export, why SQLite
isn't the source of truth here).
"""

import json
import os
import re
import shutil
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
# Finished jobs (done/ and failed/) are transient: removed this many days
# after their last update. 0 disables pruning.
RETENTION_DAYS = float(os.environ.get("LLOCKET_RETENTION_DAYS", "14"))

# A job may reference the output of jobs it depends on:
#   "{{output:<job_id>}}" anywhere in a message's content is replaced, when
#   the job is claimed, by that dependency's output content (the reply text;
#   native thinking is a separate field and never substituted).
PLACEHOLDER = re.compile(r"\{\{output:([0-9A-Za-z_-]+)\}\}")
STATES = ("queued", "running", "done", "failed")


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


def job_state(job_id: str):
    """(state, path) of a job by id, or (None, None) if it doesn't exist."""
    for state in STATES:
        path = os.path.join(QUEUE_ROOT, state, job_id)
        if os.path.isdir(path):
            return state, path
    return None, None


def placeholders_in(messages: list) -> set:
    found = set()
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, str):
            found.update(PLACEHOLDER.findall(content))
    return found


def input_chars(messages: list) -> int:
    """Characters of message text as submitted, not counting {{output:<id>}}
    placeholders (their size isn't known until the dependency finishes)."""
    total = 0
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, str):
            total += len(PLACEHOLDER.sub("", content))
    return total


def substitute_outputs(messages: list, outputs: dict) -> list:
    """Copy of messages with every {{output:<id>}} replaced by outputs[id]."""
    def repl(m):
        return outputs[m.group(1)]
    out = []
    for m in messages:
        m = dict(m)
        if isinstance(m.get("content"), str):
            m["content"] = PLACEHOLDER.sub(repl, m["content"])
        out.append(m)
    return out


def prune_finished(retention_days: float = RETENTION_DAYS, now: float = None) -> list:
    """Remove done/ and failed/ jobs whose last update is older than
    retention_days. Returns the removed job ids."""
    if retention_days <= 0:
        return []
    cutoff = (now if now is not None else time.time()) - retention_days * 86400
    removed = []
    for state in ("done", "failed"):
        root = os.path.join(QUEUE_ROOT, state)
        if not os.path.isdir(root):
            continue
        for job_id in os.listdir(root):
            path = os.path.join(root, job_id)
            try:
                updated = read_json(path, "job.json").get("updated_at")
                ts = time.mktime(time.strptime(updated, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
            except (OSError, ValueError, TypeError):
                ts = os.path.getmtime(path)
            if ts < cutoff:
                shutil.rmtree(path, ignore_errors=True)
                removed.append(job_id)
    return removed


def dependents_pending(job_id: str) -> list:
    """Queued/running jobs that depend on job_id (and so still need its output)."""
    pending = []
    for state in ("queued", "running"):
        root = os.path.join(QUEUE_ROOT, state)
        if not os.path.isdir(root):
            continue
        for other in os.listdir(root):
            try:
                deps = read_json(os.path.join(root, other), "job.json").get("depends_on") or []
            except (OSError, ValueError):
                continue
            if job_id in deps:
                pending.append(other)
    return sorted(pending)


def _scrub(path: str) -> int:
    """Best effort: overwrite a file's bytes in place with zeros and fsync
    before it's replaced or removed. On copy-on-write or snapshotting
    storage the old blocks may still survive; this narrows, not closes, the
    window. Returns the number of bytes overwritten."""
    size = os.path.getsize(path)
    with open(path, "r+b") as f:
        f.write(b"\0" * size)
        f.flush()
        os.fsync(f.fileno())
    return size


def purge_content(job_dir: str) -> dict:
    """Drop a finished job's content -- prompt messages, output text,
    thinking -- keeping its metadata (job.json, stats). Each content file is
    scrubbed (zero-overwritten, fsynced) and then replaced by a copy without
    the content, or removed. job.json records when and what was purged."""
    purged = []
    for name, keys in (("input.json", {"messages": []}),
                       ("output.json", {"content": "", "thinking": None})):
        path = os.path.join(job_dir, name)
        if not os.path.isfile(path):
            continue
        doc = read_json(job_dir, name)
        doc.update(keys)
        doc["content_purged"] = True
        nbytes = _scrub(path)
        atomic_write_json(job_dir, name, doc)
        purged.append({"file": name, "bytes_scrubbed": nbytes})
    for name in ("response.raw.jsonl",):
        path = os.path.join(job_dir, name)
        if os.path.isfile(path):
            nbytes = _scrub(path)
            os.remove(path)
            purged.append({"file": name, "bytes_scrubbed": nbytes, "removed": True})
    job = read_json(job_dir, "job.json")
    job["content_purged_at"] = now_iso()
    job["content_purged_files"] = purged
    atomic_write_json(job_dir, "job.json", job)
    return job
