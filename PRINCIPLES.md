# Llocket principles

What Llocket is and isn't. New work follows these; a change that breaks one
changes the principle here first, deliberately. Current rules only: history
lives in git.

## Scope

- **Project-agnostic.** Llocket queues chat requests and returns results.
  `project`, `purpose`, `tags` and `dedupe_key` are opaque labels the caller
  chooses; Llocket never interprets them. Nothing about a calling project
  (names, vocabulary, prompts, data, config values, history) lands in this
  repo: code, comments, docs, fixtures, examples, logs or commit messages.
- **Generalize, don't special-case.** A need that only makes sense for one
  caller becomes a parameter or a hook, or stays in that caller's repo.

## Durability

- **The directory tree is the source of truth.** A job's state is the
  directory it lives in; every transition is one atomic `rename()` of the
  whole job directory. Any index is disposable and rebuildable.
- **Never a torn file.** Every write is write-to-temp-then-rename.
- **Crash recovery is the normal path.** A restart and a crash run the same
  code: an orphaned running job is requeued with its attempt counted, or
  failed once the cap is hit.
- **A running job is never preempted.** Cancelling works on queued jobs only;
  lanes and runtime caps bound the wait instead.

## Content

- **Content is transient.** Prompts and outputs live in the queue only as
  long as a job needs them: clients purge a finished job's content once it's
  stored on their side, and retention prunes the rest.
- **Metadata is safe to show; content isn't.** Status, timing, settings,
  fingerprints and token counts may be returned and logged. Messages,
  outputs and thinking are returned only to the client that asks for that
  job's result, never in listings or logs.

## API

- **Small, explicit and backwards compatible.** New fields are added, never
  repurposed; a client that ignores a new field keeps working.
- **Say what happened.** A refusal names its reason (`409` with why), a
  superseded or cut-off job records why it ended.
- **Stdlib-only client.** `llocketclient.py` has no dependencies, so any
  project can vendor or import it.
