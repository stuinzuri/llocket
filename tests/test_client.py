import io
import json
import urllib.error

import pytest

import llocketclient as lc


class Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_submit_body_and_wait(monkeypatch):
    calls, statuses = [], iter(["queued", "running", "done"])

    def urlopen(req, timeout=None):
        calls.append((req.get_method(), req.full_url, json.loads(req.data) if req.data else None))
        if req.get_method() == "POST":
            return Resp(b'{"id": "J2", "status": "queued"}')
        return Resp(json.dumps({"id": "J2", "status": next(statuses), "output": {"content": "ok"}}).encode())
    monkeypatch.setattr(lc.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(lc.time, "sleep", lambda s: None)
    q = lc.Llocket("http://x:8000/")
    assert q.submit("p", [{"role": "user", "content": "{{output:J1}}"}], depends_on=["J1"], think=True) == "J2"
    body = calls[0][2]
    assert body["depends_on"] == ["J1"] and body["think"] is True and "model" not in body
    assert q.wait("J2", poll=0)["status"] == "done"
    assert calls[-1][1] == "http://x:8000/jobs/J2"


def test_wait_tolerates_transient_errors(monkeypatch):
    seq = [urllib.error.URLError("down"), Resp(b'{"id": "J", "status": "done"}')]

    def urlopen(req, timeout=None):
        x = seq.pop(0)
        if isinstance(x, Exception):
            raise x
        return x
    monkeypatch.setattr(lc.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(lc.time, "sleep", lambda s: None)
    polls = []
    assert lc.Llocket("http://x").wait("J", poll=0, on_poll=polls.append)["status"] == "done"
    assert polls[0]["status"] == "poll_error"


def test_http_4xx_raises(monkeypatch):
    def urlopen(req, timeout=None):
        raise urllib.error.HTTPError("u", 422, "x", {}, io.BytesIO(b'{"detail": "bad deps"}'))
    monkeypatch.setattr(lc.urllib.request, "urlopen", urlopen)
    with pytest.raises(lc.LlocketError, match="HTTP 422: bad deps"):
        lc.Llocket("http://x").submit("p", [])
