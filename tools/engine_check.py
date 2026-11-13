#!/usr/bin/env python3
"""Compare engines (or models) through the Llocket queue: the same synthetic
chat, greedy (temperature 0), thinking on and off, one job per model.

Reports prompt tokens, output tokens, generation speed and draft acceptance
for each, and whether the outputs are identical. Use it to prove a
llama-server entry formats prompts exactly like Ollama before listing a real
model in the backends file (list it under a second name first), and to
measure speed on new hardware.

    python3 tools/engine_check.py --api http://host:8000 MODEL_A MODEL_B [...]
        [--long 20000] [--max-tokens 900] [--num-ctx 65536] [--project engine-check]
        [--think both|on|off]

--long pads the conversation to roughly that many prompt tokens to measure
long-context speed (the outputs are still compared). Output text and prompt
token counts are compared separately: llama-server counts only the prompt
tokens it didn't have cached, so a lower count after an earlier job with the
same prefix is expected, not a formatting difference (rerun that case alone,
e.g. --think off, on a cold server to compare counts). Jobs run in the
standard lane at priority 10, one attempt each. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

SYSTEM = "You are a careful writing assistant. Answer in plain prose."
TURNS = [
    ("user", "Describe an old lighthouse on a rocky coast in two sentences."),
    ("assistant", "The lighthouse leans into the wind, its white paint flaking to grey stone. "
                  "Each night its lamp still sweeps the black water as if someone were watching."),
    ("user", "Now write a 150-word scene in which the keeper finds a letter in the lamp room. "
             "Keep the tone quiet and specific."),
]
FILLER = ("The harbour log for the season notes the weather each morning, the boats that left and "
          "returned, the state of the lamp, the oil delivered and the repairs made to the gallery rail. ")


def messages(long_tokens: int) -> list:
    msgs = [{"role": "system", "content": SYSTEM}]
    if long_tokens:
        n = max(1, long_tokens // 40)      # ~40 tokens per filler sentence
        msgs.append({"role": "user", "content": "Background notes:\n\n" + FILLER * n})
        msgs.append({"role": "assistant", "content": "Noted."})
    msgs += [{"role": r, "content": c} for r, c in TURNS]
    return msgs


def call(api: str, method: str, path: str, body: dict = None):
    req = urllib.request.Request(api + path, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("models", nargs="+")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--long", type=int, default=0, help="pad the prompt to about this many tokens")
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--num-ctx", type=int, default=65536, help="context for every job (Ollama's default may truncate)")
    ap.add_argument("--project", default="engine-check")
    ap.add_argument("--think", choices=("both", "on", "off"), default="both", help="which thinking modes to run")
    a = ap.parse_args()

    opts = {"temperature": 0, "seed": 1, "num_predict": a.max_tokens, "num_ctx": a.num_ctx}
    msgs = messages(a.long)
    modes = {"both": (True, False), "on": (True,), "off": (False,)}[a.think]
    jobs = {}
    for think in modes:
        for model in a.models:
            jobs[(think, model)] = call(a.api, "POST", "/jobs", {
                "project": a.project, "purpose": "engine check", "model": model, "messages": msgs,
                "think": think, "sampling_options": opts, "priority": 10, "max_attempts": 1})["id"]
    print(f"submitted {len(jobs)} jobs; waiting", file=sys.stderr)
    done = {}
    while len(done) < len(jobs):
        time.sleep(15)
        for key, jid in jobs.items():
            if key not in done:
                j = call(a.api, "GET", f"/jobs/{jid}")
                if j["status"] in ("done", "failed", "cancelled"):
                    done[key] = j
    ok = True
    for think in modes:
        print(f"\nthinking {'on' if think else 'off'}")
        ref = None
        for model in a.models:
            j = done[(think, model)]
            o = j.get("output") or {}
            if j["status"] != "done":
                print(f"  {model}: {j['status']} {j.get('error')}")
                ok = False
                continue
            tps = o["eval_count"] / (o["eval_duration_ns"] / 1e9) if o.get("eval_duration_ns") else 0
            acc = (f", drafts accepted {o['draft_n_accepted']}/{o['draft_n']}"
                   f" ({o['draft_n_accepted'] / o['draft_n']:.0%})" if o.get("draft_n") else "")
            text = (o.get("thinking") or "") + "\x00" + (o.get("content") or "")
            same = ""
            if ref is None:
                ref = (o["prompt_eval_count"], text)
            else:
                same = " | SAME output as first" if text == ref[1] else " | DIFFERENT output from first"
                if o["prompt_eval_count"] != ref[0]:
                    same += f" (prompt tokens {o['prompt_eval_count']} vs {ref[0]}: fewer can mean a cached prefix)"
                if text != ref[1]:
                    ok = False
            print(f"  {model} [{o.get('backend', '?')}]: prompt {o['prompt_eval_count']}, output {o['eval_count']},"
                  f" {tps:.2f} tok/s{acc}{same}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
