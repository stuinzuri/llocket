import json
import os

import pytest


def make_job(lib, job_id, content="hi", depends_on=None, state="queued", priority=None):
    d = os.path.join(lib.QUEUE_ROOT, state, job_id)
    os.makedirs(d)
    lib.atomic_write_json(d, "job.json", {
        "id": job_id, "project": "p", "purpose": None, "created_at": lib.now_iso(),
        "updated_at": lib.now_iso(), "status": state, "attempts": 0, "max_attempts": 3,
        "model": "m", "host": "h", "error": None, "depends_on": depends_on or [],
        **({"priority": priority} if priority is not None else {})})
    lib.atomic_write_json(d, "input.json", {"messages": [{"role": "user", "content": content}],
                                            "think": False, "sampling_options": {}})
    return d


def finish(lib, job_id, text):
    d = os.path.join(lib.QUEUE_ROOT, "queued", job_id)
    if not os.path.isdir(d):
        d = make_job(lib, job_id)
    lib.atomic_write_json(d, "output.json", {"content": text})
    os.rename(d, os.path.join(lib.QUEUE_ROOT, "done", job_id))


def test_placeholders_and_substitution(q):
    lib, _ = q
    msgs = [{"role": "system", "content": "x"}, {"role": "user", "content": "A {{output:J1}} B {{output:J2}}"}]
    assert lib.placeholders_in(msgs) == {"J1", "J2"}
    out = lib.substitute_outputs(msgs, {"J1": "one", "J2": "two"})
    assert out[1]["content"] == "A one B two" and msgs[1]["content"].startswith("A {{")


def test_waiting_job_is_skipped_and_later_one_runs(q):
    lib, worker = q
    make_job(lib, "20260101T000000Z-aaaa", "first")                 # dependency, still queued
    make_job(lib, "20260101T000001Z-bbbb", "{{output:20260101T000000Z-aaaa}}",
             depends_on=["20260101T000000Z-aaaa"])
    claimed = worker.claim_next_job()
    assert claimed.endswith("20260101T000000Z-aaaa")               # the dependency, FIFO
    os.rename(claimed, os.path.join(lib.QUEUE_ROOT, "queued", "20260101T000000Z-aaaa"))
    # with its dependency re-queued (not done), the dependent is never claimed
    os.rename(os.path.join(lib.QUEUE_ROOT, "queued", "20260101T000000Z-aaaa"),
              os.path.join(lib.QUEUE_ROOT, "running", "20260101T000000Z-aaaa"))
    assert worker.claim_next_job() is None


def test_dependency_output_substituted_at_run(q, monkeypatch):
    lib, worker = q
    finish(lib, "20260101T000000Z-aaaa", "THE OUTLINE")
    make_job(lib, "20260101T000001Z-bbbb", "Write from: {{output:20260101T000000Z-aaaa}}",
             depends_on=["20260101T000000Z-aaaa"])
    seen = {}

    def fake_call(host, model, messages, think=False, options=None):
        seen["messages"] = messages
        return {"message": {"content": "BODY", "thinking": None}, "done_reason": "stop"}
    monkeypatch.setattr(worker, "call_ollama", fake_call)
    worker.process_job(worker.claim_next_job())
    assert seen["messages"][0]["content"] == "Write from: THE OUTLINE"
    done = os.path.join(lib.QUEUE_ROOT, "done", "20260101T000001Z-bbbb")
    job = lib.read_json(done, "job.json")
    assert job["resolved_dependencies"] == {"20260101T000000Z-aaaa": len("THE OUTLINE")}
    # the stored input keeps the placeholder (the queue stays a faithful record)
    assert "{{output:" in lib.read_json(done, "input.json")["messages"][0]["content"]


def test_failed_dependency_fails_dependent(q):
    lib, worker = q
    d = make_job(lib, "20260101T000000Z-aaaa")
    os.rename(d, os.path.join(lib.QUEUE_ROOT, "failed", "20260101T000000Z-aaaa"))
    make_job(lib, "20260101T000001Z-bbbb", depends_on=["20260101T000000Z-aaaa"])
    make_job(lib, "20260101T000002Z-cccc", depends_on=["20260101T000000Z-zzzz"])
    make_job(lib, "20260101T000003Z-dddd", "independent")
    claimed = worker.claim_next_job()
    assert claimed.endswith("20260101T000003Z-dddd")
    failed = sorted(os.listdir(os.path.join(lib.QUEUE_ROOT, "failed")))
    assert failed == ["20260101T000000Z-aaaa", "20260101T000001Z-bbbb", "20260101T000002Z-cccc"]
    err = lib.read_json(os.path.join(lib.QUEUE_ROOT, "failed", "20260101T000002Z-cccc"), "job.json")["error"]
    assert "20260101T000000Z-zzzz (missing)" in err


def test_prune_finished(q):
    lib, _ = q
    finish(lib, "20200101T000000Z-old0", "x")
    job_dir = os.path.join(lib.QUEUE_ROOT, "done", "20200101T000000Z-old0")
    j = lib.read_json(job_dir, "job.json")
    j["updated_at"] = "2020-01-01T00:00:00Z"
    lib.atomic_write_json(job_dir, "job.json", j)
    finish(lib, "20990101T000000Z-new0", "y")
    make_job(lib, "20200101T000001Z-queu")
    removed = lib.prune_finished(retention_days=14)
    assert removed == ["20200101T000000Z-old0"]
    assert os.path.isdir(os.path.join(lib.QUEUE_ROOT, "queued", "20200101T000001Z-queu"))
    assert lib.prune_finished(retention_days=0) == []


def test_api_validates_dependencies(q):
    pytest.importorskip("fastapi")
    import importlib
    import api
    importlib.reload(api)
    lib, _ = q
    finish(lib, "20260101T000000Z-aaaa", "x")
    ok = api.create_job(api.JobRequest(project="p", messages=[{"role": "user", "content": "{{output:20260101T000000Z-aaaa}}"}],
                                       depends_on=["20260101T000000Z-aaaa"]))
    assert ok["status"] == "queued"
    with pytest.raises(api.HTTPException) as e:
        api.create_job(api.JobRequest(project="p", messages=[{"role": "user", "content": "{{output:X}}"}]))
    assert e.value.status_code == 422
    with pytest.raises(api.HTTPException):
        api.create_job(api.JobRequest(project="p", messages=[], depends_on=["nope"]))
    got = api.get_job(ok["id"])
    assert got.get("waiting_on") == [] and got["depends_on"] == ["20260101T000000Z-aaaa"]


def test_purge_content_keeps_meta(q):
    lib, worker = q
    finish(lib, "20260101T000000Z-aaaa", "SECRET OUTPUT")
    d = os.path.join(lib.QUEUE_ROOT, "done", "20260101T000000Z-aaaa")
    lib.atomic_write_json(d, "output.json", {"content": "SECRET OUTPUT", "thinking": "SECRET THOUGHT",
                                             "eval_count": 7})
    job = lib.purge_content(d)
    assert job["content_purged_at"]
    assert {f["file"] for f in job["content_purged_files"]} == {"input.json", "output.json"}
    assert all(f["bytes_scrubbed"] > 0 for f in job["content_purged_files"])
    blob = open(os.path.join(d, "input.json")).read() + open(os.path.join(d, "output.json")).read()
    assert "SECRET" not in blob and "hi" not in lib.read_json(d, "input.json")["messages"]
    assert lib.read_json(d, "output.json")["eval_count"] == 7
    assert lib.read_json(d, "job.json")["id"] == "20260101T000000Z-aaaa"


def test_purged_dependency_fails_dependent(q):
    lib, worker = q
    finish(lib, "20260101T000000Z-aaaa", "X")
    lib.purge_content(os.path.join(lib.QUEUE_ROOT, "done", "20260101T000000Z-aaaa"))
    make_job(lib, "20260101T000001Z-bbbb", "{{output:20260101T000000Z-aaaa}}",
             depends_on=["20260101T000000Z-aaaa"])
    worker.process_job(worker.claim_next_job())
    err = lib.read_json(os.path.join(lib.QUEUE_ROOT, "failed", "20260101T000001Z-bbbb"), "job.json")["error"]
    assert "was purged" in err and "resubmit" in err


def test_api_purge_rules(q):
    pytest.importorskip("fastapi")
    import importlib
    import api
    importlib.reload(api)
    lib, _ = q
    finish(lib, "20260101T000000Z-aaaa", "X")
    make_job(lib, "20260101T000001Z-bbbb", "{{output:20260101T000000Z-aaaa}}",
             depends_on=["20260101T000000Z-aaaa"])
    with pytest.raises(api.HTTPException) as e:
        api.purge_job("20260101T000000Z-aaaa")
    assert e.value.status_code == 409 and "20260101T000001Z-bbbb" in e.value.detail
    with pytest.raises(api.HTTPException) as e:
        api.purge_job("20260101T000001Z-bbbb")
    assert e.value.status_code == 409
    os.rename(os.path.join(lib.QUEUE_ROOT, "queued", "20260101T000001Z-bbbb"),
              os.path.join(lib.QUEUE_ROOT, "done", "20260101T000001Z-bbbb"))
    assert api.purge_job("20260101T000000Z-aaaa")["content_purged_at"]


def test_input_chars_skips_placeholders(q):
    lib, _ = q
    msgs = [{"role": "system", "content": "abc"}, {"role": "user", "content": "de{{output:J1}}f"}]
    assert lib.input_chars(msgs) == 6


def test_priority_runs_first_then_fifo_and_never_skips_dependencies(q):
    lib, worker = q
    make_job(lib, "20260101T000000Z-aaaa", "batch 1")                       # no priority: counts as 0
    make_job(lib, "20260101T000001Z-bbbb", "batch 2", priority=0)
    make_job(lib, "20260101T000002Z-cccc", "interactive", priority=10)
    make_job(lib, "20260101T000003Z-dddd", "needs batch 1", priority=20,
             depends_on=["20260101T000000Z-aaaa"])                         # not runnable yet
    order = []
    while True:
        claimed = worker.claim_next_job()
        if claimed is None:
            break
        job_id = os.path.basename(claimed)
        order.append(job_id[-4:])
        finish_dir = os.path.join(lib.QUEUE_ROOT, "done", job_id)
        lib.atomic_write_json(claimed, "output.json", {"content": "x"})
        os.rename(claimed, finish_dir)
    assert order == ["cccc", "aaaa", "dddd", "bbbb"]
