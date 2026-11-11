"""Job tags (set at submission) and run tags (added by the worker)."""
import os

import pytest

from test_identical import _api, _req


def test_tags_are_kept_deduped_and_dont_change_the_fingerprint(q):
    lib, _worker = q
    api = _api(q)
    a = api.create_job(_req(api, tags=["exp:a", " exp:a ", "", "batch"]))["id"]
    b = api.create_job(_req(api))["id"]
    ja = lib.read_json(os.path.join(lib.QUEUE_ROOT, "queued", a), "job.json")
    jb = lib.read_json(os.path.join(lib.QUEUE_ROOT, "queued", b), "job.json")
    assert ja["tags"] == ["exp:a", "batch"]
    assert ja["fingerprint"] == jb["fingerprint"]              # labels, not part of the request


def test_too_many_or_too_long_tags_are_refused(q):
    api = _api(q)
    with pytest.raises(api.HTTPException):
        api.create_job(_req(api, tags=[f"t{i}" for i in range(api.MAX_TAGS + 1)]))
    with pytest.raises(api.HTTPException):
        api.create_job(_req(api, tags=["x" * (api.MAX_TAG_LEN + 1)]))


def test_run_tags_add_host_engine_and_model_tags(q, monkeypatch):
    _lib, worker = q
    monkeypatch.setattr(worker, "HOST_TAGS", ["chip:some-chip", "rack:a"])
    job = {"tags": ["exp:a"]}
    assert worker.run_tags(job, {"backend": "llama-server", "engine_tags": ["opt:speculative"]}) == \
        ["exp:a", "chip:some-chip", "rack:a", "engine:llama-server", "opt:speculative"]
    assert worker.run_tags({}, {"backend": "ollama"}) == ["chip:some-chip", "rack:a", "engine:ollama"]


def test_finished_job_output_carries_tags(q, monkeypatch):
    lib, worker = q
    from test_dependencies import make_job
    d = make_job(lib, "20260101T000000Z-t")
    job = lib.read_json(d, "job.json")
    job["tags"] = ["exp:a"]
    lib.atomic_write_json(d, "job.json", job)
    monkeypatch.setattr(worker, "HOST_TAGS", ["chip:c"])
    monkeypatch.setattr(worker, "run_model", lambda *a, **k: {"message": {"content": "ok"}, "backend": "ollama"})
    worker.process_job(worker.claim_next_job())
    out = lib.read_json(os.path.join(lib.QUEUE_ROOT, "done", "20260101T000000Z-t"), "output.json")
    assert out["tags"] == ["exp:a", "chip:c", "engine:ollama"]


def test_chip_tag_is_a_slug():
    import worker
    tags = worker._chip_tag()
    assert all(t.startswith("chip:") and " " not in t for t in tags)
