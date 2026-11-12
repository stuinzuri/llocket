# llocket: shared, project-agnostic infrastructure

llocket is a durable async job queue for Ollama generation requests, used by several unrelated projects. It must stay project-agnostic.

## Hard rule: no bleed from calling projects

Sessions from any project may work on llocket, but nothing specific to a calling project
may land here: no project names, domain vocabulary, prompts, data, facts about a
project or its business, config values, or project history. That covers code, comments, docs,
test fixtures, example values, log messages, and commit messages. Use generic
placeholders (`project="my-project"`, `app="my-app"`). Project-specific glue
belongs in the calling project, which imports or calls llocket.

If a change only makes sense in terms of one caller's needs, generalize it (a
parameter, a hook) or keep it in that caller's repo instead.

## Principles

What this repo is and how it behaves is in PRINCIPLES.md; follow it, and change it
there first when a decision changes:

@PRINCIPLES.md
