# Llocket

<img src="assets/logo.png" alt="Llocket logo" width="96">

A durable, async job-ticket queue for Ollama generation requests -- a
llama-themed pun on "docket" (a list of cases queued for hearing, one at
a time, in order), which fits a single sequential worker processing
generation jobs that can run for tens of minutes to hours each.

Llocket is content-agnostic: it queues chat-completion requests against
a local Ollama instance and returns results, without knowing or caring
what any caller is actually generating. Any number of unrelated
projects can submit jobs against the same queue, distinguished only by
an opaque `project` tag the caller supplies.

## Design

A filesystem-as-queue design: a job's state *is* which directory its
files live in, and every state transition is a single atomic `rename()`
(POSIX and NFSv3 both guarantee this is atomic as long as source and
destination share a filesystem). SQLite is deliberately not the source
of truth here -- a networked filesystem makes SQLite's own locking
model unsafe, and a plain directory tree needs no locking at all for
this access pattern. A local index, if one exists, is treated as fully
disposable and rebuildable from the directory tree itself.

```
queue/
  .incoming/<id>/          -- being written, not yet visible to the worker
  queued/<id>/              -- job.json, input.json
  running/<id>/              -- claimed by the worker
  done/<id>/                 -- + output.json (message, token counts, timings in ns), response.raw.jsonl
  failed/<id>/               -- + error info
```

**`job.json`** (small control record, cheap to read for status polls):
`id` (timestamp-prefixed + random suffix, sorts chronologically),
`project` (caller's namespace, opaque to Llocket), `purpose` (caller's
free-text tag, also opaque), `status`, `attempts`/`max_attempts`
(capped-retry: an orphan found in `running/` at worker startup gets
requeued with `attempts += 1`, or moved to `failed/` once the cap is
hit -- crash recovery and a normal restart share the identical code
path, on purpose), `model`, `host`, `error`.

**`input.json`**: `messages`, `think` (default `False` -- leaving this
unset lets a thinking-capable model burn its entire `num_predict`
budget on reasoning tokens and emit `content: ""` with
`done_reason: "length"`, which looks like a silent failure but isn't),
`sampling_options` (passed to Ollama verbatim).

Every individual file write goes through write-to-temp-then-rename,
never an in-place overwrite -- a crash mid-write can never leave a
torn/partial file visible under its real name. Every job-state
transition moves the *whole* per-job directory in one `rename()`, never
separate renames of sibling files -- there's no window where a job's
fileset is split across two states.

**Code:** `lib.py` (shared primitives), `worker.py` (claims the oldest
queued job, runs it, handles the orphan-recovery scan at startup),
`api.py` (FastAPI: `POST /jobs`, `GET /jobs/{id}`, `GET
/jobs?project=&status=&lane=&since=`, `DELETE /jobs/{id}` -- cancel only works
pre-claim, race-checked). `submit_job.py` is manual test scaffolding
for exercising the worker without the API -- not a real integration
point. `smoke_test.py` runs the full lifecycle (create -> publish ->
claim -> call Ollama -> write output -> complete) against a real
queue and a real Ollama instance. It calls Ollama directly (not via the
worker) with a tiny model (`$LLOCKET_SMOKE_MODEL`, default `qwen3:0.6b`,
pulled once) at a small `num_ctx`, so it loads next to a busy production
model and never waits behind a long job.

## Running it

Runs under its own dedicated, non-admin service account, isolated from
any other Ollama instance on the same machine -- own binary (no package
manager, extracted directly from an official release, checksum-verified
against the release's own checksums file), own model store, own port.
This isolation means several independent Ollama instances (different
models, different callers) can coexist on one machine without
interfering with each other.

`QUEUE_ROOT` and `OLLAMA_HOST` are both overridable via environment
variable (`LLOCKET_QUEUE_ROOT`, `LLOCKET_OLLAMA_HOST`), so the worker
and API can run fully offline against a local directory and a local
Ollama instance for development, without touching a real queue.

Ollama itself runs cleanly as a boot-time LaunchDaemon on macOS -- it
only reads local model files, never a network mount, so it's unaffected
by the constraint described below.

## Dependent jobs (chains)

A job can declare `depends_on: [<job_id>, ...]` and reference a
dependency's output anywhere in its message content as
`{{output:<job_id>}}`. The worker only claims it once every dependency
is `done`, then substitutes each placeholder with that dependency's
output text (the reply content -- native thinking is a separate field
and never substituted). The stored `input.json` keeps the placeholders;
`job.json` records `resolved_dependencies` (id -> substituted length).

- A job waiting on dependencies stays in `queued/` (the API reports
  `waiting_on`); runnable jobs behind it are not blocked.
- If a dependency fails (or is missing), the dependent fails straight
  away with `dependency failed or missing: ...`.
- The API rejects placeholders that aren't declared in `depends_on`,
  unknown dependencies (422) and already-failed ones (409).

This lets a client submit a whole multi-step chain up front and
disconnect: the chain runs to completion server-side.

## Priority

A job can set `priority` (an integer, default 0). The worker takes the
highest-priority runnable job first, and the oldest first within a
priority -- so short interactive jobs can be submitted at, say, 10 and
run ahead of a long batch queued at 0. Nothing is preempted: a running
job always finishes first. A waiting dependent job keeps its own priority
(it doesn't inherit its dependency's). There is no aging, so a steady
stream of high-priority jobs can hold back low-priority ones; keep high
priorities for short, occasional work.

## Lanes

Each job runs in a lane: `"standard"` (default) or `"backlog"`.

- **Standard** jobs run in priority order, as above.
- **Backlog** jobs are for low-urgency work nobody is waiting on
  (keeping derived results up to date, for example). One runs only when no
  standard job is runnable, whatever its priority, so it never delays a
  standard job that is already queued. Among backlog jobs, priority and FIFO
  order apply as usual.
- **Runtime cap.** A running job is never preempted, so a standard job
  submitted while a backlog job runs waits for it. To bound that wait,
  every backlog job is cut off after `max_runtime_s` (default and maximum
  `LLOCKET_BACKLOG_MAX_RUNTIME_S`, 600s). A cut-off job fails with
  `exceeded max runtime: ...` and isn't retried. Keep backlog jobs well
  under the cap. A standard job may set `max_runtime_s` too; it has no
  limit by default.
- **`dedupe_key`.** When a job is submitted with a `dedupe_key`, any
  queued job of the same project with the same key is cancelled
  (`superseded by <new id>`) and the new one takes its place. Running and
  finished jobs are left alone, and so is a queued job something else
  depends on. A client can resubmit the same piece of work whenever its
  inputs change without piling up stale copies. Works in either lane.

`GET /jobs?lane=backlog` lists one lane. `job.json` records `lane`,
`max_runtime_s` (the effective limit, or null) and `dedupe_key`.

## Client

`llocketclient.py` -- stdlib-only client + CLI (`submit`, `job`, `jobs`,
`cancel`, `wait`; `python3 llocketclient.py --api URL list|status|wait|cancel`).
`wait()` tolerates transient network errors; the job never depends on
the client staying connected.

## Job timing

The worker stamps `started_at` into `job.json` when it picks a job up, so a
client can tell how long the running job has been going (`created_at` is
when it was submitted; `updated_at` when it last changed state).
`input_chars` is the size of the submitted message text (placeholders
not counted) and `input_placeholders` how many `{{output:<id>}}`
placeholders it has, so a client can estimate how long a job will take
before it starts. Jobs submitted before these fields existed lack them.

## Purging content

`POST /jobs/{id}/purge` (client: `purge()`) empties a finished job's
content -- the prompt messages in `input.json`, the output text and
thinking in `output.json` -- and stamps `content_purged_at` in
`job.json`; status, timings and token counts stay. Meant for clients to
call as soon as they've safely stored a result, so sensitive text sits
in the queue only while a job is in flight. Refused (409) while the job
is queued/running or while a pending job still depends on its output. A
dependent that runs after its dependency was purged fails with a clear
error naming when it was purged.

Each content file is scrubbed first (zero-overwritten in place and
fsynced), then replaced or removed. That's best effort: on copy-on-write
or snapshotting storage the old blocks can survive until the storage
reclaims them. `job.json` records `content_purged_at` and
`content_purged_files` (file, bytes scrubbed); the API logs every purge
and every refused purge (with the reason) to its stdout log.

## Retention

Finished jobs are transient: the worker removes `done/` and `failed/`
jobs `LLOCKET_RETENTION_DAYS` (default 14) days after their last update,
checked hourly. `0` disables pruning. Queued/running jobs are never
pruned.

## A macOS constraint worth knowing

**Resolved 2026-09-28: granting Full Disk Access (System Settings ->
Privacy & Security) to `/usr/bin/python3` *and* to the Command Line
Tools `Python.app` it hands off to lets the worker and API run as
boot-time LaunchDaemons with the queue on NFS.** Both entries were
needed; the bundle alone was not enough (launchd starts
`/usr/bin/python3`). Trade-off: every script run with that interpreter
gets Full Disk Access -- acceptable on a dedicated service machine.
The investigation below predates that finding and is kept for context.

Also mount the share *soft* with a dead-server timeout (e.g.
`soft,intr,timeo=30,retrans=3,deadtimeout=60`): a default (hard) NFS
mount turns any NAS/network hiccup into processes stuck in the kernel
that only a reboot clears.

Original finding: **LaunchDaemons and LaunchAgents cannot access an
NFS mount, in any configuration tried -- macOS's TCC privacy system
working as designed.**

### What doesn't work

Every one of these produces an identical `PermissionError: [Errno 1]
Operation not permitted` on a plain `os.listdir()` of the mounted
directory:

1. NFS mounted by a non-root user (no `resvport`)
2. NFS mounted by root (no `resvport`)
3. NFS mounted by root **with** `resvport` (a "secure"-flagged mount,
   no `nodev`/`nosuid`)
4. The mount established via `autofs` (`/etc/auto_master` + an autofs
   map) instead of an ad hoc `mount_nfs` call -- macOS's own natively
   integrated automounting mechanism
5. The worker/API run as a `system`-domain LaunchDaemon
6. The identical scripts run as a `gui/<uid>`-domain LaunchAgent under
   an already-active, already-logged-in session
7. A non-Apple-signed Python interpreter in place of the platform
   binary `/usr/bin/python3`

None of it matters. Every interactive SSH session, by contrast -- any
account, freshly connected, no session reuse -- can read/write the same
directory every time without issue, ruling out a stale/cached-session
explanation.

### The actual mechanism

The unified log's TCC subsystem shows the real story. The relevant
service is `kTCCServiceSystemPolicyNetworkVolumes` -- macOS's "Network
Volumes" privacy protection category. Two things are both true:

- **A bare command-line binary with no `.app` bundle/Info.plist cannot
  hold its own independent TCC grant.** Its access is always attributed
  through whichever *responsible* GUI app invoked it. Granting access
  interactively (running the command from a real terminal app, clicking
  Allow on the resulting system prompt) records the grant against the
  terminal app as the responsible party, with the interpreter binary
  merely "accessing" under no bundle identity of its own -- never
  against the interpreter binary standing alone.
- **launchd is not a GUI app and cannot be a "responsible" party.**
  Whatever launchd spawns directly -- LaunchDaemon or LaunchAgent,
  doesn't matter -- has no responsible app in its attribution chain at
  all, so there is no possible grant for it to inherit, regardless of
  which Python interpreter runs it or whether an interactive grant was
  ever given elsewhere. A platform binary gets an automatic, by-design
  denial for unattended requests, no prompt ever shown; a non-platform
  binary is denied too, just for a different reason (no attributable
  grantor exists at all).

This also explains why a loopback-SSH connection (`ssh <user>@localhost`)
empirically works around the block: an authenticated login session
appears to get treated with the same standing as a real interactive
session for this specific check.

A loopback-SSH LaunchAgent (a service that SSHes into itself on every
boot using a key it trusts for itself) was considered as a workaround
and deliberately rejected: a standing self-trusted key plus a service
that auto-connects to itself is structurally identical to a
persistence/backdoor mechanism. If that key or its trust were ever
tampered with, it's a ready-made way back in -- not a trade worth making
for convenience.

### Real options

1. **A remote-supervised process**: a separate, already-trusted machine
   keeps a persistent SSH session open to the service account, running
   the worker/API inside it. Avoids creating any new self-trust
   relationship, at the cost of making the service's uptime depend on
   that separate machine being on and connected.
2. **A properly bundled, code-signed helper app** (a real `.app`,
   `Info.plist`, stable bundle identifier) that can hold its own
   independent TCC grant rather than borrowing a terminal app's. Likely
   the correct long-term fix, but real packaging/signing investment.
3. **Manual/supervised operation**: run the worker/API in a plain
   terminal session (`screen`/`tmux`) kept alive by hand -- not
   reboot-resilient, but simple and immediately available.

## Setup scripts

All scripts live in `setup/` and are parameterized -- no hostnames,
usernames, or paths are hardcoded; substitute your own values wherever
a script's usage comment or a plist's `YOUR_*` placeholder calls for
one.

- **`install_ollama.sh [version] [port] [base_dir]`** -- downloads,
  checksum-verifies, and extracts the Ollama binary; writes and loads a
  per-user LaunchAgent. Useful for a first interactive test; the daemon
  plist below is the production pattern.
- **`copy_known_good_models.sh <src_models_dir> <dst_models_dir>
  <manifest-path>...`** -- copies just the blobs/manifests a given set
  of models needs from another Ollama store on the same machine,
  skipping blobs already present at the destination and verifying every
  copy by hash. Generic -- not tied to any specific model list.
- **`com.llocket.ollama.daemon.plist`** -- the LaunchDaemon definition
  for running Ollama itself as a boot-time service under a dedicated
  account. Fill in the `YOUR_USERNAME` placeholders, then install as
  root: `sudo cp` it into `/Library/LaunchDaemons/`, `sudo chown
  root:wheel`, `sudo chmod 644`, `sudo launchctl bootstrap system
  <path>`.
- **`mount_llocket_nas.sh <nas_host> <nas_export> <mount_point>`** +
  `auto_llocket` (an `autofs` map for the same mount, referenced from
  `/etc/auto_master`) -- idempotent NFS mounting for a networked queue
  directory, with retries in case the network isn't up yet at boot.
- **`com.llocket.nas-mount.plist`** / **`com.llocket.worker.plist`** /
  **`com.llocket.api.plist`** -- LaunchAgent/LaunchDaemon definitions
  for the mount script, worker, and API respectively. The worker/API
  ones are not currently deployable as boot-time services, per the TCC
  constraint above -- present here as the straightforward (non-
  workaround) definitions for whenever that has a real resolution.

Verify Ollama's up: `curl http://<host>:<port>/api/version` and
`/api/tags`. Verify the API (when running): `curl
http://<host>:8000/jobs`.

## Prior art

The serving layer (Ollama) and multi-provider gateways (LiteLLM and
similar) are solved problems, but the specific niche here -- one
sequential worker, tens-of-minutes-to-hours jobs, full durable
input/output history -- isn't covered by a mature off-the-shelf tool.
[InferHub](https://github.com/Dev-Art-Solutions/InferHub) is the
closest match but is a multi-node GPU-worker mesh design, a different
shape than a single sequential worker. Its job-status pattern (`POST
.../jobs` -> id, `GET events` over SSE for queued -> running ->
succeeded transitions) is worth a look for API-shape ideas even though
not adopted here.

Revisit self-developing this versus adopting something else if the
project ever grows toward: multiple GPU nodes, high job-per-second
throughput (past a plain filesystem's comfort zone), or enough general
interest to justify spinning it out further.

## Known limitations

The job-queue REST API and worker run as boot-time LaunchDaemons under
the service account (see "A macOS constraint worth knowing" for the
Full Disk Access grant this needs). Single sequential worker by design.
