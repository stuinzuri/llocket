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


def test_run_tags_carry_the_engine_version(q, monkeypatch):
    _lib, worker = q
    monkeypatch.setattr(worker, "HOST_TAGS", [])
    assert worker.run_tags({}, {"backend": "ollama", "engine_version": "ollama-1.2.3"}) == \
        ["engine:ollama", "engine-version:ollama-1.2.3"]
    assert worker.run_tags({}, {"backend": "ollama", "engine_version": None}) == ["engine:ollama"]


def test_llama_server_version_comes_from_the_binary():
    import engines
    fake = os.path.join(os.path.dirname(__file__), "fake-llama-server")
    assert engines.llama_server_version(fake) == "b4321"
    assert engines.engine_version("llama-server", "http://127.0.0.1:1", fake) == "llama-server-b4321"
    assert engines.llama_server_version("/no/such/binary") is None


def test_llama_server_version_reads_the_newer_build_format(monkeypatch, tmp_path):
    import subprocess
    import engines
    b = tmp_path / "llama-server"
    b.write_text("")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: type("P", (), {
        "stdout": "", "stderr": "version: 0.9.0-dev (build 12284, commit 9e9619d03)\nbuilt with a compiler\n"})())
    assert engines.llama_server_version(str(b)) == "b12284"


def test_ollama_version_is_none_when_it_does_not_answer():
    import engines
    assert engines.ollama_version("http://127.0.0.1:1", timeout=0.5) is None
    assert engines.engine_version("ollama", "http://127.0.0.1:1") is None


def test_get_job_shows_settings_never_messages(q):
    api = _api(q)
    jid = api.create_job(_req(api, think=True, sampling_options={"temperature": 0.7}))["id"]
    job = api.get_job(jid)
    assert job["settings"] == {"think": True, "sampling_options": {"temperature": 0.7}}
    assert "messages" not in job and "messages" not in job["settings"]
