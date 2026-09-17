# Llocket

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
  done/<id>/                 -- + output.json, response.raw.jsonl
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
/jobs?project=&status=`, `DELETE /jobs/{id}` -- cancel only works
pre-claim, race-checked). `submit_job.py` is manual test scaffolding
for exercising the worker without the API -- not a real integration
point. `smoke_test.py` runs the full lifecycle (create -> publish ->
claim -> call Ollama -> write output -> complete) against a real
queue and a real Ollama instance.

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

## A macOS constraint worth knowing

**LaunchDaemons and LaunchAgents cannot access an NFS mount, in any
configuration -- this is macOS's TCC privacy system working as
designed, not a bug or a configuration mistake.**

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

The job-queue REST API and worker are built and validated end-to-end
(normal completion, crash recovery, capped-retry-to-failed, and the
full REST round trip) but are **not currently running as boot-time
services** -- see "A macOS constraint worth knowing" above. Ollama
itself is the only piece of this that's actually daemonized reliably
right now.
