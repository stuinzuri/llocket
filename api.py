#!/usr/bin/env python3
"""Llocket's REST API -- the real replacement for submit_job.py's manual
testing shape. A separate process from worker.py on purpose: the API only
ever writes into queued/ (via .incoming/ + one atomic rename) or reads
from the queue directories; it never touches running/ itself and has no
idea whether a worker is even alive. The two processes are decoupled
through the filesystem, exactly like everything else in this design.

    POST   /jobs        enqueue a job, returns its id
    GET    /jobs/{id}   status + result (searches all four state dirs)
    GET    /jobs         list jobs, optional ?project=&status= filters
    DELETE /jobs/{id}   cancel -- only possible while still queued

Run under the account that owns the queue and Ollama instance:
    python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
"""

import os
import random
import string
import time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from lib import OLLAMA_HOST, QUEUE_ROOT, atomic_write_json, ensure_queue_dirs, now_iso, read_json

DEFAULT_MODEL = "orcarouter/Qwen3.8-27B-Uncensored:q8_0"
STATES = ("queued", "running", "done", "failed")

app = FastAPI(title="Llocket")


class JobRequest(BaseModel):
    project: str
    messages: list
    model: str = DEFAULT_MODEL
    host: str = OLLAMA_HOST
    purpose: Optional[str] = None
    think: bool = False
    sampling_options: Optional[dict] = None
    max_attempts: int = 3


def new_job_id() -> str:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suffix = "".join(random.choices(string.hexdigits.lower(), k=8))
    return f"{ts}-{suffix}"


def find_job_dir(job_id: str):
    """Returns (state, path) for a job id by checking each state dir --
    the queue is small enough (single sequential worker, no high job
    volume) that this is fine as a plain existence check per state rather
    than needing an index. Returns (None, None) if not found anywhere."""
    for state in STATES:
        path = os.path.join(QUEUE_ROOT, state, job_id)
        if os.path.isdir(path):
            return state, path
    return None, None


@app.on_event("startup")
def _startup():
    ensure_queue_dirs()


@app.post("/jobs")
def create_job(req: JobRequest):
    job_id = new_job_id()
    job = {
        "id": job_id,
        "project": req.project,
        "purpose": req.purpose,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "status": "queued",
        "attempts": 0,
        "max_attempts": req.max_attempts,
        "model": req.model,
        "host": req.host,
        "error": None,
    }
    input_doc = {
        "messages": req.messages,
        "think": req.think,
        "sampling_options": req.sampling_options or {},
    }

    incoming_dir = os.path.join(QUEUE_ROOT, ".incoming", job_id)
    os.makedirs(incoming_dir)
    atomic_write_json(incoming_dir, "job.json", job)
    atomic_write_json(incoming_dir, "input.json", input_doc)

    queued_dir = os.path.join(QUEUE_ROOT, "queued", job_id)
    os.rename(incoming_dir, queued_dir)  # atomic publish

    return {"id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    state, path = find_job_dir(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="job not found")

    job = read_json(path, "job.json")
    result = dict(job)
    # Directory location is authoritative, not job.json's own "status" field
    # -- the worker only rewrites that field at completion, not at claim
    # time, so a running job's job.json can still say "queued" for its
    # whole runtime otherwise.
    result["status"] = state
    if state == "done":
        result["output"] = read_json(path, "output.json")
    return result


@app.get("/jobs")
def list_jobs(project: Optional[str] = None, status: Optional[str] = None):
    states_to_scan = (status,) if status else STATES
    jobs = []
    for state in states_to_scan:
        state_dir = os.path.join(QUEUE_ROOT, state)
        if not os.path.isdir(state_dir):
            continue
        for job_id in sorted(os.listdir(state_dir)):
            try:
                job = read_json(os.path.join(state_dir, job_id), "job.json")
            except (FileNotFoundError, ValueError):
                continue  # mid-write or genuinely corrupt -- skip, don't crash a list call
            if project and job.get("project") != project:
                continue
            job["status"] = state  # directory location is authoritative, see get_job()
            jobs.append(job)
    return jobs


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str):
    state, path = find_job_dir(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail="job not found")
    if state != "queued":
        raise HTTPException(
            status_code=409,
            detail=f"cannot cancel a job in state '{state}' -- only 'queued' jobs can be cancelled",
        )

    job = read_json(path, "job.json")
    job["status"] = "failed"
    job["error"] = "cancelled"
    job["updated_at"] = now_iso()
    atomic_write_json(path, "job.json", job)

    failed_dir = os.path.join(QUEUE_ROOT, "failed", job_id)
    try:
        os.rename(path, failed_dir)
    except FileNotFoundError:
        # The worker claimed it (queued/ -> running/) in the gap between our
        # check above and this rename -- too late to cancel now.
        raise HTTPException(status_code=409, detail="job was claimed by the worker before it could be cancelled")
    return {"id": job_id, "status": "failed", "error": "cancelled"}
