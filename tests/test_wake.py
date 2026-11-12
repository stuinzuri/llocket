import importlib
import os
import shutil
import tempfile
import threading
import time

import pytest

import llocketclient as lc


@pytest.fixture
def short_sock(monkeypatch):
    d = tempfile.mkdtemp(prefix="lw")   # AF_UNIX paths are short: keep it out of pytest's deep tmp dirs
    path = os.path.join(d, "w.sock")
    monkeypatch.setenv("LLOCKET_WAKE_SOCKET", path)
    yield path
    shutil.rmtree(d, ignore_errors=True)


def _api(q):
    pytest.importorskip("fastapi")
    import api
    importlib.reload(api)
    return api


def test_submit_wakes_an_idle_worker_at_once(q, short_sock):
    lib, _ = q
    importlib.reload(lib)
    listener = lib.WakeListener()
    api = _api(q)
    t0 = time.monotonic()
    threading.Timer(0.1, lambda: api.create_job(api.JobRequest(project="p", messages=[]))).start()
    assert listener.wait(5) is True
    assert time.monotonic() - t0 < 2
    assert listener.wait(0.05) is False   # pings were drained


def test_wake_without_a_listener_is_harmless(q, short_sock):
    lib, _ = q
    importlib.reload(lib)
    assert lib.wake_worker() is False
    assert lib.WakeListener(path="").wait(0.01) is False


def test_get_job_long_polls_until_the_state_changes(q):
    lib, _ = q
    api = _api(q)
    job_id = api.create_job(api.JobRequest(project="p", messages=[]))["id"]
    src = os.path.join(lib.QUEUE_ROOT, "queued", job_id)
    threading.Timer(0.3, lambda: os.rename(src, os.path.join(lib.QUEUE_ROOT, "running", job_id))).start()
    t0 = time.monotonic()
    assert api.get_job(job_id, wait=5, status="queued")["status"] == "running"
    assert 0.2 < time.monotonic() - t0 < 2
    t0 = time.monotonic()
    assert api.get_job(job_id, wait=5, status="queued")["status"] == "running"   # already moved on: at once
    assert time.monotonic() - t0 < 0.2


def test_client_wait_long_polls_with_the_last_seen_state(monkeypatch):
    seen, statuses = [], iter(["queued", "running", "done"])

    def job(self, job_id, wait=0, status=None):
        seen.append((wait, status))
        return {"id": job_id, "status": next(statuses)}
    monkeypatch.setattr(lc.Llocket, "job", job)
    slept = []
    monkeypatch.setattr(lc.time, "sleep", slept.append)
    assert lc.Llocket("http://x").wait("J", poll=15)["status"] == "done"
    assert seen == [(0, None), (15, "queued"), (15, "running")]
    assert slept == []   # every change came back straight away


def test_client_wait_falls_back_to_polling_on_an_older_server(monkeypatch):
    statuses = iter(["queued", "queued", "done"])
    monkeypatch.setattr(lc.Llocket, "job", lambda self, j, wait=0, status=None: {"id": j, "status": next(statuses)})
    slept = []
    monkeypatch.setattr(lc.time, "sleep", slept.append)
    assert lc.Llocket("http://x").wait("J", poll=15)["status"] == "done"
    assert len(slept) == 1 and slept[0] > 14   # answered at once with the same state: waited out the poll
