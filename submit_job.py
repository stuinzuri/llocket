#!/usr/bin/env python3
"""Manual job submission for testing worker.py -- not the real API (that's
still to be built), just enough to publish a job into queued/ by hand.

Usage: python3 submit_job.py "some prompt" [model]
"""

import os
import random
import string
import sys
import time

from lib import OLLAMA_HOST, QUEUE_ROOT, atomic_write_json, ensure_queue_dirs, now_iso

DEFAULT_MODEL = "orcarouter/Qwen3.8-27B-Uncensored:q8_0"


def new_job_id() -> str:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suffix = "".join(random.choices(string.hexdigits.lower(), k=8))
    return f"{ts}-{suffix}"


def submit(prompt: str, model: str = DEFAULT_MODEL, project: str = "worker-test") -> str:
    ensure_queue_dirs()
    job_id = new_job_id()

    job = {
        "id": job_id,
        "project": project,
        "purpose": "manual_test",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "status": "queued",
        "attempts": 0,
        "max_attempts": 3,
        "model": model,
        "host": OLLAMA_HOST,
        "error": None,
    }
    input_doc = {
        "messages": [{"role": "user", "content": prompt}],
        "sampling_options": {"num_predict": 40, "temperature": 0.2},
    }

    incoming_dir = os.path.join(QUEUE_ROOT, ".incoming", job_id)
    os.makedirs(incoming_dir)
    atomic_write_json(incoming_dir, "job.json", job)
    atomic_write_json(incoming_dir, "input.json", input_doc)

    queued_dir = os.path.join(QUEUE_ROOT, "queued", job_id)
    os.rename(incoming_dir, queued_dir)
    print(job_id)
    return job_id


if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Reply with exactly the words: worker test ok"
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    submit(prompt, model)
