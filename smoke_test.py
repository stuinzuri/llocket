#!/usr/bin/env python3
"""Smoke test for Llocket's job-lifecycle mechanism -- NOT the real worker.

Proves the actual pieces work together for real, on this box, against the
real NAS mount and the real Ollama instance:

    create (in .incoming/) -> publish (queued/) -> claim (running/)
    -> call Ollama -> write output -> complete (done/)

Run under the account that owns the queue and Ollama instance:
    python3 ~/llocket/smoke_test.py
"""

import json
import os
import random
import string
import sys
import time
import urllib.error

from lib import (
    OLLAMA_HOST,
    atomic_write_json,
    call_ollama,
    ensure_queue_dirs,
    now_iso,
    QUEUE_ROOT,
)

MODEL = "orcarouter/Qwen3.8-27B-Uncensored:q8_0"
PROJECT = "smoke-test"


def new_job_id() -> str:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suffix = "".join(random.choices(string.hexdigits.lower(), k=8))
    return f"{ts}-{suffix}"


def main() -> int:
    ensure_queue_dirs()

    job_id = new_job_id()
    print(f"[{job_id}] creating job")

    job = {
        "id": job_id,
        "project": PROJECT,
        "purpose": "smoke_test",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "status": "queued",
        "attempts": 0,
        "max_attempts": 3,
        "model": MODEL,
        "host": OLLAMA_HOST,
        "error": None,
    }
    input_doc = {
        "messages": [
            {"role": "user", "content": "Reply with exactly the words: llocket smoke test ok"}
        ],
        "sampling_options": {"num_predict": 40, "temperature": 0.2},
    }

    # --- create: write fully in a not-yet-visible staging dir ---
    incoming_dir = os.path.join(QUEUE_ROOT, ".incoming", job_id)
    os.makedirs(incoming_dir)
    atomic_write_json(incoming_dir, "job.json", job)
    atomic_write_json(incoming_dir, "input.json", input_doc)

    # --- publish: one atomic directory rename makes it visible in queued/ ---
    queued_dir = os.path.join(QUEUE_ROOT, "queued", job_id)
    os.rename(incoming_dir, queued_dir)
    print(f"[{job_id}] published to queued/")

    # --- claim: move the whole job directory into running/ ---
    running_dir = os.path.join(QUEUE_ROOT, "running", job_id)
    os.rename(queued_dir, running_dir)
    print(f"[{job_id}] claimed into running/")

    # --- execute against the real Ollama instance ---
    print(f"[{job_id}] calling Ollama ({MODEL}) -- model isn't resident, this includes cold load...")
    start = time.time()
    try:
        result = call_ollama(
            OLLAMA_HOST, MODEL, input_doc["messages"],
            options=input_doc["sampling_options"],
        )
    except (urllib.error.URLError, TimeoutError) as e:
        job["status"] = "failed"
        job["error"] = f"ollama call failed: {e}"
        job["updated_at"] = now_iso()
        atomic_write_json(running_dir, "job.json", job)
        failed_dir = os.path.join(QUEUE_ROOT, "failed", job_id)
        os.rename(running_dir, failed_dir)
        print(f"[{job_id}] FAILED: {e}")
        return 1
    elapsed = time.time() - start
    print(f"[{job_id}] Ollama responded in {elapsed:.1f}s")

    # --- write output (still inside running/, still temp-then-rename) ---
    output_doc = {
        "content": result.get("message", {}).get("content", ""),
        "done_reason": result.get("done_reason"),
        "prompt_eval_count": result.get("prompt_eval_count"),
        "eval_count": result.get("eval_count"),
        "total_duration_ns": result.get("total_duration"),
    }
    atomic_write_json(running_dir, "output.json", output_doc)

    job["status"] = "done"
    job["updated_at"] = now_iso()
    atomic_write_json(running_dir, "job.json", job)

    # --- complete: one atomic directory rename into done/ ---
    done_dir = os.path.join(QUEUE_ROOT, "done", job_id)
    os.rename(running_dir, done_dir)
    print(f"[{job_id}] completed -> done/")

    # --- verify: read back what a real caller would read ---
    with open(os.path.join(done_dir, "job.json")) as f:
        final_job = json.load(f)
    with open(os.path.join(done_dir, "output.json")) as f:
        final_output = json.load(f)

    print(f"[{job_id}] readback status={final_job['status']} done_reason={final_output['done_reason']}")
    print(f"[{job_id}] content={final_output['content']!r}")

    if final_job["status"] != "done" or not final_output["content"]:
        print(f"[{job_id}] SMOKE TEST FAILED: unexpected final state")
        return 1

    print(f"[{job_id}] SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
