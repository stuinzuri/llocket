import importlib

import pytest


def _api(q):
    pytest.importorskip("fastapi")
    import api
    importlib.reload(api)
    return api


def _req(api, **kw):
    base = dict(project="p", messages=[{"role": "user", "content": "hi"}], purpose="x")
    return api.JobRequest(**{**base, **kw})


def test_identical_jobs_join_only_when_asked(q):
    api = _api(q)
    first = api.create_job(_req(api))["id"]
    copy = api.create_job(_req(api))                               # not asked: a second job
    assert copy["id"] != first
    joined = api.create_job(_req(api, join_identical=True))
    assert joined == {"id": first, "status": "queued", "joined": True}
    assert api.get_job(first)["fingerprint"] == api.get_job(copy["id"])["fingerprint"]


def test_any_difference_makes_a_new_job(q):
    api = _api(q)
    first = api.create_job(_req(api))["id"]
    for change in (dict(messages=[{"role": "user", "content": "hi!"}]), dict(sampling_options={"temperature": 0.5}),
                   dict(priority=5), dict(lane="backlog"), dict(model="other"), dict(project="q"), dict(think=True)):
        assert api.create_job(_req(api, join_identical=True, **change))["id"] != first, change


def test_cancel_drops_one_claim_at_a_time(q):
    api = _api(q)
    job_id = api.create_job(_req(api))["id"]
    api.create_job(_req(api, join_identical=True))
    assert api.cancel_job(job_id)["status"] == "queued"           # the joiner still wants it
    assert api.cancel_job(job_id)["status"] == "failed"


def test_jobs_queued_before_fingerprints_still_match(q):
    lib, _ = q
    api = _api(q)
    import os
    job_id = api.create_job(_req(api))["id"]
    path = os.path.join(lib.QUEUE_ROOT, "queued", job_id)
    job = lib.read_json(path, "job.json")
    job.pop("fingerprint")
    lib.atomic_write_json(path, "job.json", job)
    assert api.create_job(_req(api, join_identical=True))["id"] == job_id
    assert [j["fingerprint"] for j in api.list_jobs(status="queued")][0]
