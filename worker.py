#!/usr/bin/env python3
"""Llocket's worker loop: claims queued jobs one at a time, runs them
against the local Ollama instance, and files the result -- durable via
the NAS-backed filesystem-as-queue mechanism (see lib.py and README.md
for the design).

Deliberately no signal handling. A kill/crash mid-job just leaves that
job's directory sitting in running/ -- exactly what the startup recovery
scan below is for. Normal shutdown and a crash are handled by the same
path on purpose, rather than adding a separate graceful-shutdown branch
that would only ever be exercised on purpose, not by an actual failure.

Single sequential worker, FIFO by job id (timestamp-prefixed, so lexical
sort is chronological) -- no message broker, one worker loop is the
whole queue.

Run under the account that owns the queue and Ollama instance:
    python3 ~/llocket/worker.py
"""

from __future__ import annotations  # this account runs Python 3.9; needed for `str | None`

import os
import sys
import time
import urllib.error

from lib import (
    OLLAMA_HOST,
    QUEUE_ROOT,
    atomic_write_json,
    call_ollama,
    ensure_queue_dirs,
    now_iso,
    read_json,
)

POLL_INTERVAL_SECONDS = 5


def _requeue_or_fail(job_dir: str, job: dict, reason: str) -> None:
    """Shared by startup orphan-recovery and live job failures: bump the
    attempt count, and either send the job back to queued/ for another
    try or, past max_attempts, to failed/ for good. Same policy either
    way -- a crash-recovered job and a live failure are both just
    "this attempt didn't finish," not different kinds of event."""
    job_id = job["id"]
    job["attempts"] += 1
    job["error"] = reason
    job["updated_at"] = now_iso()

    if job["attempts"] >= job["max_attempts"]:
        job["status"] = "failed"
        job["error"] = f"exceeded retry cap ({job['max_attempts']} attempts): {reason}"
        atomic_write_json(job_dir, "job.json", job)
        dest = os.path.join(QUEUE_ROOT, "failed", job_id)
        print(f"[{job_id}] attempts={job['attempts']}/{job['max_attempts']} -> failed/: {reason}")
    else:
        job["status"] = "queued"
        atomic_write_json(job_dir, "job.json", job)
        dest = os.path.join(QUEUE_ROOT, "queued", job_id)
        print(f"[{job_id}] attempts={job['attempts']}/{job['max_attempts']} -> requeued: {reason}")

    os.rename(job_dir, dest)


def recover_orphans() -> None:
    """Run once at startup (every startup, not just after a known crash --
    recovery is the normal path, not a rare branch). Anything sitting in
    running/ was mid-flight when whatever last ran this worker stopped
    without finishing it."""
    running_root = os.path.join(QUEUE_ROOT, "running")
    orphans = sorted(os.listdir(running_root))
    if not orphans:
        print("startup: no orphaned jobs in running/")
        return

    print(f"startup: found {len(orphans)} orphaned job(s) in running/, recovering")
    for job_id in orphans:
        job_dir = os.path.join(running_root, job_id)
        try:
            job = read_json(job_dir, "job.json")
        except (FileNotFoundError, ValueError) as e:
            # Can't even read the record -- nothing safe to do but leave it
            # for a human rather than guess. Doesn't block the rest of the
            # queue.
            print(f"[{job_id}] unreadable job.json ({e}), leaving in running/ for manual review")
            continue
        _requeue_or_fail(job_dir, job, "orphaned in running/ (worker restarted or crashed mid-job)")


def claim_next_job() -> str | None:
    """Returns the path to a freshly-claimed job directory in running/, or
    None if queued/ is empty. FIFO by job id."""
    queued_root = os.path.join(QUEUE_ROOT, "queued")
    candidates = sorted(os.listdir(queued_root))
    for job_id in candidates:
        src = os.path.join(queued_root, job_id)
        dst = os.path.join(QUEUE_ROOT, "running", job_id)
        try:
            os.rename(src, dst)
            return dst
        except FileNotFoundError:
            # Only matters if something else claims jobs from this same
            # queue concurrently -- not true today (single worker), kept
            # cheap and harmless in case that ever changes.
            continue
    return None


def process_job(job_dir: str) -> None:
    job = read_json(job_dir, "job.json")
    job_id = job["id"]
    input_doc = read_json(job_dir, "input.json")

    print(f"[{job_id}] running ({job['model']})")
    start = time.time()
    try:
        result = call_ollama(
            job["host"],
            job["model"],
            input_doc["messages"],
            think=input_doc.get("think", False),
            options=input_doc.get("sampling_options"),
        )
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        _requeue_or_fail(job_dir, job, f"ollama call failed: {e}")
        return
    elapsed = time.time() - start
    print(f"[{job_id}] responded in {elapsed:.1f}s, done_reason={result.get('done_reason')}")

    output_doc = {
        "content": result.get("message", {}).get("content", ""),
        "thinking": result.get("message", {}).get("thinking"),
        "done_reason": result.get("done_reason"),
        "prompt_eval_count": result.get("prompt_eval_count"),
        "eval_count": result.get("eval_count"),
        "total_duration_ns": result.get("total_duration"),
    }
    atomic_write_json(job_dir, "output.json", output_doc)

    job["status"] = "done"
    job["updated_at"] = now_iso()
    atomic_write_json(job_dir, "job.json", job)

    done_dir = os.path.join(QUEUE_ROOT, "done", job_id)
    os.rename(job_dir, done_dir)
    print(f"[{job_id}] -> done/")


def main() -> int:
    ensure_queue_dirs()
    recover_orphans()

    print(f"worker started, polling every {POLL_INTERVAL_SECONDS}s, host={OLLAMA_HOST}")
    while True:
        job_dir = claim_next_job()
        if job_dir is None:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        try:
            process_job(job_dir)
        except Exception as e:  # noqa: BLE001 -- a bug in one job must never kill the worker
            print(f"unexpected error processing {job_dir}: {e}")
            try:
                job = read_json(job_dir, "job.json")
                _requeue_or_fail(job_dir, job, f"unexpected worker error: {e}")
            except Exception as inner:
                print(f"could not even recover {job_dir}: {inner} -- leaving for manual review")


if __name__ == "__main__":
    sys.exit(main())
