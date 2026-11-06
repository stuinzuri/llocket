#!/usr/bin/env python3
"""Llocket's REST API -- the real replacement for submit_job.py's manual
testing shape. A separate process from worker.py on purpose: the API only
ever writes into queued/ (via .incoming/ + one atomic rename) or reads
from the queue directories; it never touches running/ itself and has no
idea whether a worker is even alive. The two processes are decoupled
through the filesystem, exactly like everything else in this design.

    POST   /jobs        enqueue a job, returns its id (optional depends_on +
                        {{output:<id>}} placeholders, see lib.PLACEHOLDER)
    GET    /jobs/{id}   status + result (searches all four state dirs)
    GET    /jobs         list jobs, optional ?project=&status=&lane= filters
    DELETE /jobs/{id}   cancel -- only possible while still queued
    POST   /jobs/{id}/purge  drop a finished job's content (prompt, output,
                        thinking), keeping its metadata -- for clients that
                        have safely stored the result and want nothing
                        sensitive left in the queue

Run under the account that owns the queue and Ollama instance:
    python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
"""

import os
import random
import string
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from lib import (LANES, OLLAMA_HOST, QUEUE_ROOT, STATES, atomic_write_json, dependents_pending, effective_max_runtime,
                 ensure_queue_dirs, input_chars, job_state, now_iso, placeholders_in, purge_content, read_json)

DEFAULT_MODEL = "orcarouter/Qwen3.8-27B-Uncensored:q8_0"

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
    depends_on: List[str] = []
    priority: int = 0          # higher runs first among runnable jobs; see README "Priority"
    lane: str = "standard"     # "standard" or "backlog"; see README "Lanes"
    max_runtime_s: Optional[float] = None   # cut the call off after this long (backlog: always, capped)
    dedupe_key: Optional[str] = None        # replaces this project's queued job with the same key


def new_job_id() -> str:
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suffix = "".join(random.choices(string.hexdigits.lower(), k=8))
    return f"{ts}-{suffix}"


def find_job_dir(job_id: str):
    """Returns (state, path) for a job id, or (None, None) if not found."""
    return job_state(job_id)


def waiting_on(job: dict) -> list:
    return [d for d in job.get("depends_on") or [] if job_state(d)[0] != "done"]


@app.on_event("startup")
def _startup():
    ensure_queue_dirs()


def _supersede(project: str, dedupe_key: str, new_id: str) -> list:
    """Cancel this project's queued jobs with the same dedupe_key (the new
    job replaces them). Running and finished jobs are left alone. Returns
    the ids cancelled."""
    queued_root = os.path.join(QUEUE_ROOT, "queued")
    gone = []
    for job_id in sorted(os.listdir(queued_root)):
        path = os.path.join(queued_root, job_id)
        try:
            job = read_json(path, "job.json")
        except (FileNotFoundError, ValueError):
            continue
        if job.get("project") != project or job.get("dedupe_key") != dedupe_key:
            continue
        if dependents_pending(job_id):
            continue   # something queued is waiting on its output
        job.update(status="failed", error=f"superseded by {new_id}", updated_at=now_iso())
        try:
            atomic_write_json(path, "job.json", job)
            os.rename(path, os.path.join(QUEUE_ROOT, "failed", job_id))
            gone.append(job_id)
        except (FileNotFoundError, OSError):
            continue   # claimed by the worker meanwhile: it runs, the new job runs too
    return gone


@app.post("/jobs")
def create_job(req: JobRequest):
    if req.lane not in LANES:
        raise HTTPException(status_code=422, detail=f"lane must be one of {list(LANES)}")
    if req.max_runtime_s is not None and req.max_runtime_s <= 0:
        raise HTTPException(status_code=422, detail="max_runtime_s must be positive")
    deps = list(dict.fromkeys(req.depends_on))
    refs = placeholders_in(req.messages)
    undeclared = sorted(refs - set(deps))
    if undeclared:
        raise HTTPException(status_code=422, detail=f"placeholders reference jobs not in depends_on: {undeclared}")
    for d in deps:
        state, _ = job_state(d)
        if state is None:
            raise HTTPException(status_code=422, detail=f"dependency {d} not found")
        if state == "failed":
            raise HTTPException(status_code=409, detail=f"dependency {d} already failed")
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
        "depends_on": deps,
        "priority": req.priority,
        "lane": req.lane,
        "max_runtime_s": effective_max_runtime(req.lane, req.max_runtime_s),
        "dedupe_key": req.dedupe_key,
        # prompt size, for clients estimating run time (see README "Job timing")
        "input_chars": input_chars(req.messages),
        "input_placeholders": len(refs),
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
    superseded = _supersede(req.project, req.dedupe_key, job_id) if req.dedupe_key else []
    os.rename(incoming_dir, queued_dir)  # atomic publish

    return {"id": job_id, "status": "queued", **({"superseded": superseded} if superseded else {})}


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
    if state == "queued" and job.get("depends_on"):
        result["waiting_on"] = waiting_on(job)
    if state == "done":
        result["output"] = read_json(path, "output.json")
    return result


@app.get("/jobs")
def list_jobs(project: Optional[str] = None, status: Optional[str] = None, lane: Optional[str] = None):
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
            if lane and (job.get("lane") or "standard") != lane:
                continue
            job["status"] = state  # directory location is authoritative, see get_job()
            if state == "queued" and job.get("depends_on"):
                job["waiting_on"] = waiting_on(job)
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


@app.post("/jobs/{job_id}/purge")
def purge_job(job_id: str):
    state, path = find_job_dir(job_id)
    if state is None:
        print(f"[{job_id}] purge refused: job not found", flush=True)
        raise HTTPException(status_code=404, detail="job not found")
    if state not in ("done", "failed"):
        print(f"[{job_id}] purge refused: job is {state}", flush=True)
        raise HTTPException(status_code=409, detail=f"job is {state}; only finished jobs can be purged")
    pending = dependents_pending(job_id)
    if pending:
        print(f"[{job_id}] purge refused: still needed by {', '.join(pending)}", flush=True)
        raise HTTPException(status_code=409, detail=f"still needed by pending job(s): {', '.join(pending)}")
    job = purge_content(path)
    files = ", ".join(f"{f['file']} ({f['bytes_scrubbed']}B)" for f in job["content_purged_files"])
    print(f"[{job_id}] purged content ({state}): {files or 'no content files'}", flush=True)
    return {"id": job_id, "status": state, "content_purged_at": job["content_purged_at"],
            "content_purged_files": job["content_purged_files"]}
