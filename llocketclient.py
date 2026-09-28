"""Llocket client -- stdlib only (Python 3.9+), usable from any project.

    from llocketclient import Llocket
    q = Llocket("http://llocket-host:8000")
    a = q.submit("my-project", [{"role": "user", "content": "Write a haiku."}], purpose="draft")
    b = q.submit("my-project",
                 [{"role": "user", "content": "Improve this haiku:\\n{{output:" + a + "}}"}],
                 depends_on=[a], purpose="revise")
    job = q.wait(b)            # blocks, polling; tolerates transient network errors
    print(job["output"]["content"])

Jobs run on the server whether or not this client stays connected: submit,
disconnect, and come back later with job()/wait(). A job may depend on
earlier jobs and reference their output text with "{{output:<job_id>}}";
the worker runs it only after they're done and substitutes their output.

CLI:
    python3 llocketclient.py [--api URL] list [--project P] [--status S]
    python3 llocketclient.py [--api URL] status <job_id>
    python3 llocketclient.py [--api URL] wait <job_id>
    python3 llocketclient.py [--api URL] cancel <job_id>
    python3 llocketclient.py [--api URL] purge <job_id>
$LLOCKET_API sets the default API URL.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, List, Optional

DEFAULT_API = os.environ.get("LLOCKET_API", "http://localhost:8000")
FINAL_STATES = ("done", "failed")


class LlocketError(RuntimeError):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


class Llocket:
    def __init__(self, api: str = DEFAULT_API, timeout: float = 30.0):
        self.api = api.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: Optional[dict] = None, params: Optional[dict] = None):
        url = self.api + path
        if params:
            url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read() or b"null")
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            try:
                detail = json.loads(raw).get("detail", raw)
            except ValueError:
                detail = raw
            raise LlocketError(f"HTTP {e.code}: {detail}", e.code) from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LlocketError(f"cannot reach Llocket at {self.api}: {e}") from None

    def submit(self, project: str, messages: list, *, model: Optional[str] = None,
               purpose: Optional[str] = None, think: bool = False,
               sampling_options: Optional[dict] = None, depends_on: Optional[List[str]] = None,
               max_attempts: int = 3) -> str:
        body = {"project": project, "messages": messages, "purpose": purpose, "think": think,
                "sampling_options": sampling_options or {}, "depends_on": depends_on or [],
                "max_attempts": max_attempts}
        if model:
            body["model"] = model
        return self._request("POST", "/jobs", body)["id"]

    def job(self, job_id: str) -> dict:
        return self._request("GET", f"/jobs/{job_id}")

    def jobs(self, project: Optional[str] = None, status: Optional[str] = None) -> list:
        return self._request("GET", "/jobs", params={"project": project, "status": status})

    def cancel(self, job_id: str) -> dict:
        return self._request("DELETE", f"/jobs/{job_id}")

    def purge(self, job_id: str) -> dict:
        """Drop a finished job's content on the server (prompt, output,
        thinking), keeping its metadata. Call once the result is safely
        stored client-side. 409 while it's unfinished or still needed by a
        pending dependent."""
        return self._request("POST", f"/jobs/{job_id}/purge")

    def wait(self, job_id: str, poll: float = 15.0, timeout: Optional[float] = None,
             on_poll: Optional[Callable[[dict], None]] = None) -> dict:
        """Poll until the job is done or failed; returns the final job dict.
        Transient network/HTTP-5xx errors are reported via on_poll and retried;
        the job keeps running server-side regardless."""
        start = time.monotonic()
        while True:
            try:
                job = self.job(job_id)
            except LlocketError as e:
                if e.status and e.status < 500:
                    raise
                job = {"id": job_id, "status": "poll_error", "error": str(e)}
            if on_poll:
                on_poll(job)
            if job.get("status") in FINAL_STATES:
                return job
            if timeout is not None and time.monotonic() - start > timeout:
                raise LlocketError(f"timed out after {timeout}s waiting for {job_id} (still {job.get('status')})")
            time.sleep(poll)


def _summary(job: dict) -> str:
    extra = ""
    if job.get("waiting_on"):
        extra = f" waiting_on={','.join(job['waiting_on'])}"
    if job.get("error"):
        extra += f" error={job['error']}"
    if job.get("content_purged_at"):
        extra += f" content_purged_at={job['content_purged_at']}"
    return f"{job['id']} {job.get('status')} {job.get('project')}/{job.get('purpose')}{extra}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Llocket client")
    ap.add_argument("--api", default=DEFAULT_API)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("list")
    s.add_argument("--project")
    s.add_argument("--status", choices=["queued", "running", "done", "failed"])
    for name in ("status", "wait", "cancel", "purge"):
        sub.add_parser(name).add_argument("job_id")
    args = ap.parse_args(argv)
    q = Llocket(args.api)
    try:
        if args.cmd == "list":
            for j in q.jobs(args.project, args.status):
                print(_summary(j))
        elif args.cmd == "status":
            print(_summary(q.job(args.job_id)))
        elif args.cmd == "wait":
            print(_summary(q.wait(args.job_id, on_poll=lambda j: print(_summary(j), file=sys.stderr))))
        elif args.cmd == "cancel":
            print(q.cancel(args.job_id))
        elif args.cmd == "purge":
            print(q.purge(args.job_id))
    except LlocketError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
