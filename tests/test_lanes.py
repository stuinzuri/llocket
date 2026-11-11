import importlib
import os

import pytest

from test_dependencies import make_job


def _lane(lib, job_id, lane, **extra):
    d = os.path.join(lib.QUEUE_ROOT, "queued", job_id)
    job = lib.read_json(d, "job.json")
    job.update(lane=lane, **extra)
    lib.atomic_write_json(d, "job.json", job)


def _drain(lib, worker):
    order = []
    while True:
        claimed = worker.claim_next_job()
        if claimed is None:
            return order
        job_id = os.path.basename(claimed)
        order.append(job_id[-4:])
        lib.atomic_write_json(claimed, "output.json", {"content": "x"})
        os.rename(claimed, os.path.join(lib.QUEUE_ROOT, "done", job_id))


def test_backlog_runs_only_after_every_runnable_standard_job(q):
    lib, worker = q
    make_job(lib, "20260101T000000Z-bk01", "old backlog", priority=50)   # priority doesn't lift it
    _lane(lib, "20260101T000000Z-bk01", "backlog")
    make_job(lib, "20260101T000001Z-std1", "batch")
    make_job(lib, "20260101T000002Z-bk02", "backlog")
    _lane(lib, "20260101T000002Z-bk02", "backlog")
    make_job(lib, "20260101T000003Z-std2", "interactive", priority=10)
    assert _drain(lib, worker) == ["std2", "std1", "bk01", "bk02"]


def test_backlog_runs_while_standard_waits_on_a_dependency(q):
    lib, worker = q
    os.makedirs(os.path.join(lib.QUEUE_ROOT, "running", "20260101T000000Z-dep0"))
    make_job(lib, "20260101T000001Z-std1", "needs dep", depends_on=["20260101T000000Z-dep0"])
    make_job(lib, "20260101T000002Z-bk01", "backlog")
    _lane(lib, "20260101T000002Z-bk01", "backlog")
    assert os.path.basename(worker.claim_next_job()).endswith("bk01")


def test_effective_max_runtime(q):
    lib, _ = q
    cap = lib.BACKLOG_MAX_RUNTIME_S
    assert lib.effective_max_runtime("backlog") == lib.BACKLOG_DEFAULT_RUNTIME_S < cap
    assert lib.effective_max_runtime("backlog", 120) == 120
    assert lib.effective_max_runtime("backlog", cap) == cap
    assert lib.effective_max_runtime("backlog", cap * 10) == cap
    assert lib.effective_max_runtime("standard") is None
    assert lib.effective_max_runtime("standard", 90) == 90


def test_runtime_limit_fails_without_retry(q, monkeypatch):
    lib, worker = q
    make_job(lib, "20260101T000000Z-bk01", "slow")
    _lane(lib, "20260101T000000Z-bk01", "backlog", max_runtime_s=5)
    seen = {}

    def fake_call(host, model, messages, think=False, options=None, max_runtime=None):
        seen["max_runtime"] = max_runtime
        raise lib.RuntimeLimitExceeded("stopped after 5s")
    monkeypatch.setattr(worker, "call_ollama", fake_call)
    worker.process_job(worker.claim_next_job())
    assert seen["max_runtime"] == 5
    job = lib.read_json(os.path.join(lib.QUEUE_ROOT, "failed", "20260101T000000Z-bk01"), "job.json")
    assert job["error"].startswith("exceeded max runtime") and job["attempts"] == 0


def test_standard_job_without_a_limit_gets_none(q, monkeypatch):
    lib, worker = q
    make_job(lib, "20260101T000000Z-std1", "normal")
    seen = {}

    def fake_call(host, model, messages, think=False, options=None, **kw):
        seen.update(kw)
        return {"message": {"content": "ok", "thinking": None}, "done_reason": "stop"}
    monkeypatch.setattr(worker, "call_ollama", fake_call)
    worker.process_job(worker.claim_next_job())
    assert "max_runtime" not in seen


def _api(q):
    pytest.importorskip("fastapi")
    import api
    importlib.reload(api)
    return api


def test_api_lane_validation_and_recorded_fields(q):
    api = _api(q)
    lib, _ = q
    with pytest.raises(api.HTTPException) as e:
        api.create_job(api.JobRequest(project="p", messages=[], lane="someday"))
    assert e.value.status_code == 422
    ok = api.create_job(api.JobRequest(project="p", messages=[], lane="backlog"))
    job = api.get_job(ok["id"])
    assert job["lane"] == "backlog" and job["max_runtime_s"] == lib.BACKLOG_DEFAULT_RUNTIME_S
    std = api.get_job(api.create_job(api.JobRequest(project="p", messages=[]))["id"])
    assert std["lane"] == "standard" and std["max_runtime_s"] is None
    assert [j["id"] for j in api.list_jobs(lane="backlog")] == [ok["id"]]


def test_dedupe_key_replaces_queued_job_in_same_project_only(q):
    api = _api(q)
    lib, _ = q
    first = api.create_job(api.JobRequest(project="p", messages=[], lane="backlog", dedupe_key="thing-1"))
    other = api.create_job(api.JobRequest(project="other", messages=[], lane="backlog", dedupe_key="thing-1"))
    second = api.create_job(api.JobRequest(project="p", messages=[], lane="backlog", dedupe_key="thing-1"))
    assert second["superseded"] == [first["id"]]
    gone = api.get_job(first["id"])
    assert gone["status"] == "failed" and gone["error"] == f"superseded by {second['id']}"
    assert api.get_job(other["id"])["status"] == "queued"
    assert api.get_job(second["id"])["status"] == "queued"


def test_dedupe_leaves_running_jobs_alone(q):
    api = _api(q)
    lib, worker = q
    first = api.create_job(api.JobRequest(project="p", messages=[], dedupe_key="k"))
    worker.claim_next_job()                                  # first is now running
    second = api.create_job(api.JobRequest(project="p", messages=[], dedupe_key="k"))
    assert "superseded" not in second
    assert api.get_job(first["id"])["status"] == "running"


def test_call_ollama_stops_streaming_past_max_runtime(q, monkeypatch):
    lib, _ = q
    clock = {"t": 0.0}
    monkeypatch.setattr(lib.time, "monotonic", lambda: clock["t"])

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __iter__(self):
            for i in range(100):
                clock["t"] += 10                       # each chunk "takes" 10s
                yield b'{"message": {"content": "x"}}\n'

    monkeypatch.setattr(lib.urllib.request, "urlopen", lambda req, timeout: Stream())
    with pytest.raises(lib.RuntimeLimitExceeded):
        lib.call_ollama("http://h", "m", [], max_runtime=35)
    clock["t"] = 0.0
    out = lib.call_ollama("http://h", "m", [])                # no limit: reads the whole stream
    assert out["message"]["content"] == "x" * 100


def test_list_jobs_since_skips_older_jobs_without_reading_them(q):
    api = _api(q)
    lib, _ = q
    import os, json
    old = "20200101T000000Z-aaaaaaaa"
    os.makedirs(os.path.join(lib.QUEUE_ROOT, "done", old))
    with open(os.path.join(lib.QUEUE_ROOT, "done", old, "job.json"), "w") as f:
        json.dump({"id": old, "project": "p"}, f)
    new = api.create_job(api.JobRequest(project="p", messages=[]))["id"]
    assert {j["id"] for j in api.list_jobs()} == {old, new}
    assert [j["id"] for j in api.list_jobs(since="2021-01-01T00:00:00Z")] == [new]
    with pytest.raises(api.HTTPException):
        api.list_jobs(since="yesterday")


def _model(lib, job_id, model, age_s=0):
    d = os.path.join(lib.QUEUE_ROOT, "queued", job_id)
    job = lib.read_json(d, "job.json")
    import time
    job.update(model=model, created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - age_s)))
    lib.atomic_write_json(d, "job.json", job)


def test_last_model_runs_first_within_a_priority(q):
    lib, worker = q
    for i, (m, prio) in enumerate([("a", 0), ("b", 0), ("a", 0), ("b", 0), ("b", 10)]):
        make_job(lib, f"20260101T00000{i}Z-j{i}{m}{prio:02d}", priority=prio)
        _model(lib, f"20260101T00000{i}Z-j{i}{m}{prio:02d}", m)
    worker._last_model = ("h", "a")
    # priority still wins (the b job at 10 first); within priority 0, a's jobs before b's, FIFO within each
    assert worker._queue_order(os.path.join(lib.QUEUE_ROOT, "queued"))[0].endswith("j4b10")
    order = [j[-5:] for j in worker._queue_order(os.path.join(lib.QUEUE_ROOT, "queued"))]
    assert order == ["j4b10", "j0a00", "j2a00", "j1b00", "j3b00"]


def test_affinity_yields_once_a_job_has_waited_too_long(q):
    lib, worker = q
    make_job(lib, "20260101T000000Z-old0b")
    _model(lib, "20260101T000000Z-old0b", "b", age_s=worker.AFFINITY_MAX_WAIT_S + 60)
    make_job(lib, "20260101T000001Z-new1a")
    _model(lib, "20260101T000001Z-new1a", "a")
    worker._last_model = ("h", "a")
    assert [j[-5:] for j in worker._queue_order(os.path.join(lib.QUEUE_ROOT, "queued"))] == ["old0b", "new1a"]
    worker._last_model = None            # nothing run yet: plain FIFO
    assert [j[-5:] for j in worker._queue_order(os.path.join(lib.QUEUE_ROOT, "queued"))] == ["old0b", "new1a"]


def test_process_job_remembers_its_model(q, monkeypatch):
    lib, worker = q
    make_job(lib, "20260101T000000Z-x")
    monkeypatch.setattr(worker, "run_model", lambda *a, **k: {"message": {"content": "ok"}, "done_reason": "stop"})
    worker.process_job(worker.claim_next_job())
    assert worker._last_model == ("h", "m")
