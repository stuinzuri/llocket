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

Single sequential worker: standard lane before backlog, then priority,
then FIFO by job id (timestamp-prefixed, so lexical sort is chronological)
-- no message broker, one worker loop is the whole queue.

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
    RETENTION_DAYS,
    RuntimeLimitExceeded,
    atomic_write_json,
    call_ollama,
    effective_max_runtime,
    ensure_queue_dirs,
    job_state,
    now_iso,
    prune_finished,
    read_json,
    substitute_outputs,
)
import llama_server

POLL_INTERVAL_SECONDS = 5
PRUNE_INTERVAL_SECONDS = 3600


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


def output_doc(result: dict) -> dict:
    """What a finished job keeps from Ollama's /api/chat response: the
    message plus its counts and timings (nanoseconds, as Ollama reports
    them). eval_duration is generation time alone, so tokens per second is
    eval_count / eval_duration."""
    return {
        "content": result.get("message", {}).get("content", ""),
        "thinking": result.get("message", {}).get("thinking"),
        "done_reason": result.get("done_reason"),
        "prompt_eval_count": result.get("prompt_eval_count"),
        "eval_count": result.get("eval_count"),
        "total_duration_ns": result.get("total_duration"),
        "load_duration_ns": result.get("load_duration"),
        "prompt_eval_duration_ns": result.get("prompt_eval_duration"),
        "eval_duration_ns": result.get("eval_duration"),
        # which engine ran it, and speculative-decoding counts when it was used
        **{k: result[k] for k in ("backend", "draft_n", "draft_n_accepted") if result.get(k) is not None},
    }


_manager = None
_backends_mtime = -1.0


def manager():
    """The llama-server manager, or None when no backends file configures one.
    The file is re-read when it changes (or disappears), so a model can be
    moved between engines without restarting the worker."""
    global _manager, _backends_mtime
    try:
        mtime = os.stat(llama_server.BACKENDS_FILE).st_mtime
    except OSError:
        mtime = 0.0
    if mtime != _backends_mtime:
        if _manager:
            _manager.stop()
        cfg = llama_server.load_config() if mtime else {}
        _manager = llama_server.Manager(cfg, OLLAMA_HOST) if cfg.get("models") else None
        _backends_mtime = mtime
        if _manager:
            print(f"llama-server backend for: {', '.join(sorted(cfg['models']))}")
    return _manager


def run_model(job: dict, messages: list, input_doc: dict, max_runtime) -> dict:
    """Run the chat on llama-server when the backends file lists the model
    (default host, text only), else on Ollama. If llama-server can't start,
    the job falls back to Ollama rather than failing."""
    mgr = manager()
    options = input_doc.get("sampling_options") or {}
    think = input_doc.get("think", False)
    if mgr and job["host"] == OLLAMA_HOST and mgr.handles(job["model"]) and llama_server.supports(messages):
        try:
            loaded = mgr.ensure(job["model"], options.get("num_ctx") or 0)
            if loaded:
                print(f"[{job['id']}] llama-server started in {loaded:.0f}s")
        except (llama_server.BackendUnavailable, OSError, ValueError) as e:
            print(f"[{job['id']}] llama-server unavailable ({e}); using ollama")
            mgr.stop()
        else:
            mc = mgr.model_config(job["model"])
            body, ignored = llama_server.build_request(messages, think, options, mc.get("defaults"),
                                                       mc.get("chat_template_kwargs"))
            if ignored:
                print(f"[{job['id']}] options not supported by llama-server, ignored: {', '.join(ignored)}")
            try:
                result = llama_server.call(mgr.port, body, **({"max_runtime": max_runtime} if max_runtime else {}))
            finally:
                mgr.touch()
            result["backend"] = "llama-server"
            return result
    if mgr:
        mgr.stop()  # free its memory before Ollama loads a model
    result = call_ollama(job["host"], job["model"], messages, think=think, options=options or None,
                         **({"max_runtime": max_runtime} if max_runtime else {}))
    result["backend"] = "ollama"
    return result


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


def _fail_unrunnable(job_dir: str, job: dict, reason: str) -> None:
    job["status"] = "failed"
    job["error"] = reason
    job["updated_at"] = now_iso()
    atomic_write_json(job_dir, "job.json", job)
    os.rename(job_dir, os.path.join(QUEUE_ROOT, "failed", job["id"]))
    print(f"[{job['id']}] -> failed/: {reason}")


def _queue_order(queued_root: str) -> list:
    """Queued job ids: standard lane before backlog, then highest priority
    first, then FIFO by job id (ids start with their UTC timestamp). A job
    without a lane is standard; without a priority, 0."""
    keyed = []
    for job_id in os.listdir(queued_root):
        try:
            job = read_json(os.path.join(queued_root, job_id), "job.json")
            prio = int(job.get("priority") or 0)
            backlog = job.get("lane") == "backlog"
        except (FileNotFoundError, ValueError, TypeError):
            prio, backlog = 0, False
        keyed.append((backlog, -prio, job_id))
    return [job_id for _b, _p, job_id in sorted(keyed)]


def claim_next_job() -> str | None:
    """Returns the path to a freshly-claimed job directory in running/, or
    None if nothing is runnable. Highest priority first, FIFO by job id
    within a priority, among *runnable* jobs: a job whose dependencies
    aren't all done yet is skipped (it waits, later jobs may pass it); a
    job with a failed or missing dependency is failed straight away, so a
    broken chain never blocks the queue. A running job is never preempted."""
    queued_root = os.path.join(QUEUE_ROOT, "queued")
    for job_id in _queue_order(queued_root):
        src = os.path.join(queued_root, job_id)
        try:
            deps = read_json(src, "job.json").get("depends_on") or []
        except (FileNotFoundError, ValueError):
            continue
        if deps:
            states = {d: job_state(d)[0] for d in deps}
            bad = [d for d, st in states.items() if st in (None, "failed")]
            if bad:
                try:
                    job = read_json(src, "job.json")
                    _fail_unrunnable(src, job, "dependency failed or missing: " + ", ".join(
                        f"{d} ({states[d] or 'missing'})" for d in bad))
                except (FileNotFoundError, ValueError, OSError):
                    pass
                continue
            if any(st != "done" for st in states.values()):
                continue  # still waiting on a dependency
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


def resolve_dependencies(job: dict, messages: list) -> list:
    """Substitute {{output:<id>}} placeholders with the dependencies' output
    content. Records what was substituted in job['resolved_dependencies']."""
    deps = job.get("depends_on") or []
    if not deps:
        return messages
    outputs = {}
    for d in deps:
        state, path = job_state(d)
        if state != "done":
            raise RuntimeError(f"dependency {d} is {state or 'missing'} at claim time")
        out = read_json(path, "output.json")
        if out.get("content_purged"):
            purged_at = read_json(path, "job.json").get("content_purged_at", "?")
            raise RuntimeError(f"dependency {d}'s output was purged (at {purged_at}) before this job "
                               "ran -- its client purged it too early; resubmit the dependency")
        outputs[d] = out.get("content", "")
    job["resolved_dependencies"] = {d: len(v) for d, v in outputs.items()}
    return substitute_outputs(messages, outputs)


def process_job(job_dir: str) -> None:
    job = read_json(job_dir, "job.json")
    job_id = job["id"]
    input_doc = read_json(job_dir, "input.json")

    print(f"[{job_id}] running ({job['model']})")
    # when the worker picked it up -- lets clients tell how long it's been running
    job["started_at"] = now_iso()
    atomic_write_json(job_dir, "job.json", job)
    try:
        messages = resolve_dependencies(job, input_doc["messages"])
    except (RuntimeError, OSError, ValueError, KeyError) as e:
        _fail_unrunnable(job_dir, job, f"could not resolve dependencies: {e}")
        return
    start = time.time()
    max_runtime = effective_max_runtime(job.get("lane") or "standard", job.get("max_runtime_s"))
    try:
        result = run_model(job, messages, input_doc, max_runtime)
    except RuntimeLimitExceeded as e:
        # not retried: another attempt would take just as long
        _fail_unrunnable(job_dir, job, f"exceeded max runtime: {e}")
        return
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        _requeue_or_fail(job_dir, job, f"model call failed: {e}")
        return
    elapsed = time.time() - start
    print(f"[{job_id}] responded in {elapsed:.1f}s, done_reason={result.get('done_reason')}")

    atomic_write_json(job_dir, "output.json", output_doc(result))

    job["status"] = "done"
    job["updated_at"] = now_iso()
    atomic_write_json(job_dir, "job.json", job)

    done_dir = os.path.join(QUEUE_ROOT, "done", job_id)
    os.rename(job_dir, done_dir)
    print(f"[{job_id}] -> done/")


def main() -> int:
    ensure_queue_dirs()
    recover_orphans()
    if manager() and manager().reap_stale():
        print("stopped a llama-server left running by a previous worker")

    print(f"worker started, polling every {POLL_INTERVAL_SECONDS}s, host={OLLAMA_HOST}, "
          f"retention {RETENTION_DAYS} days")
    last_prune = 0.0
    while True:
        if time.time() - last_prune > PRUNE_INTERVAL_SECONDS:
            last_prune = time.time()
            try:
                removed = prune_finished()
                if removed:
                    print(f"pruned {len(removed)} finished job(s) older than {RETENTION_DAYS} days")
            except OSError as e:
                print(f"prune failed: {e}")
        job_dir = claim_next_job()
        if job_dir is None:
            if manager() and manager().idle_tick():
                print("llama-server idle, stopped")
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
