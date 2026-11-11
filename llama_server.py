"""Optional llama-server backend: serve chosen models with llama.cpp's own
server instead of Ollama, so they can use engine options Ollama doesn't
expose (e.g. speculative decoding with the model's built-in MTP layer).

Off unless a backends file exists (LLOCKET_BACKENDS_FILE, default
~/llocket/backends.json). The file lives on the host, never in this repo:

    {"llama_server": {
        "binary": "/path/to/llama-server",
        "port": 11438,
        "idle_stop_s": 600,
        "models": {
            "<ollama model name>": {
                "gguf": "/path/to/model.gguf",
                "ctx": 65536,
                "args": ["--spec-type", "draft-mtp"],
                "defaults": {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
            }}}}

A job for a listed model (on the default Ollama host, text only) runs on
llama-server; everything else runs on Ollama exactly as before. Callers
can't tell the difference: the request is translated from Ollama's
/api/chat shape and the reply mapped back to it.

Sampling parity: every sampler setting is sent explicitly, so
llama-server's own defaults never apply. The precedence is Ollama's
built-in defaults, then the model's "defaults" (copy them from
`ollama show --parameters`), then the job's sampling_options -- the same
order Ollama uses.

Memory: one backend holds a model at a time. Starting llama-server first
unloads every model Ollama has resident; an Ollama job first stops
llama-server. An idle llama-server stops after idle_stop_s. The worker
owns the process (pid file beside the backends file, so a restarted worker
cleans up one it left behind)."""

from __future__ import annotations  # this account runs Python 3.9

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request

BACKENDS_FILE = os.environ.get("LLOCKET_BACKENDS_FILE", os.path.expanduser("~/llocket/backends.json"))

# Ollama's built-in sampler defaults (what it sends its runner when neither the
# model's parameters nor the job set a value). llama-server's differ (min_p 0.05,
# top_p 0.95, ...), so these are always sent.
OLLAMA_DEFAULTS = {"temperature": 0.8, "top_k": 40, "top_p": 0.9, "min_p": 0.0, "typical_p": 1.0,
                   "repeat_penalty": 1.0, "repeat_last_n": 64, "presence_penalty": 0.0,
                   "frequency_penalty": 0.0}
# Ollama option names llama-server accepts under the same name.
PASSTHROUGH = set(OLLAMA_DEFAULTS) | {"seed", "stop", "mirostat", "mirostat_tau", "mirostat_eta"}
# Ollama options that are about the runner, not sampling: handled here or not applicable.
RUNNER_OPTIONS = {"num_ctx", "num_predict", "num_gpu", "num_thread", "num_batch", "use_mmap", "keep_alive"}


class BackendUnavailable(Exception):
    """llama-server couldn't be started; the caller should use Ollama instead."""


def load_config(path: str = BACKENDS_FILE) -> dict:
    """The llama_server section of the backends file, or {} if there is none."""
    try:
        with open(path) as f:
            return json.load(f).get("llama_server") or {}
    except FileNotFoundError:
        return {}


def build_request(messages: list, think: bool, options: dict = None, defaults: dict = None) -> tuple:
    """(body for /v1/chat/completions, option names that were ignored)."""
    options = dict(options or {})
    params = {**OLLAMA_DEFAULTS, **(defaults or {})}
    ignored = []
    for k, v in options.items():
        if k in PASSTHROUGH:
            params[k] = v
        elif k not in RUNNER_OPTIONS:
            ignored.append(k)
    body = {"messages": messages, "stream": True, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": bool(think)}, **params}
    n = options.get("num_predict")
    if n is not None and int(n) > 0:
        body["max_tokens"] = int(n)
    return body, sorted(ignored)


def supports(messages: list) -> bool:
    """Text-only chats; anything with images stays on Ollama."""
    return not any(m.get("images") for m in messages)


def call(port: int, body: dict, stall_timeout: float = 600.0, max_runtime: float = None) -> dict:
    """Stream a chat from llama-server and return it shaped like Ollama's
    non-streaming /api/chat reply (durations in ns), plus draft_n /
    draft_n_accepted when speculative decoding ran. Raises the same
    exceptions as lib.call_ollama for the worker to handle the same way."""
    from lib import RuntimeLimitExceeded
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    started = time.monotonic()
    timeout = min(stall_timeout, max_runtime) if max_runtime else stall_timeout
    content, thinking, finish, timings = [], [], None, {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                if max_runtime and time.monotonic() - started > max_runtime:
                    raise RuntimeLimitExceeded(f"stopped after {max_runtime:.0f}s (the job's max_runtime_s)")
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if "error" in obj:
                    raise OSError(f"llama-server stream error: {obj['error']}")
                for ch in obj.get("choices") or []:
                    delta = ch.get("delta") or {}
                    content.append(delta.get("content") or "")
                    thinking.append(delta.get("reasoning_content") or "")
                    finish = ch.get("finish_reason") or finish
                timings = obj.get("timings") or timings
    except OSError as e:  # URLError and TimeoutError are OSErrors
        if max_runtime and time.monotonic() - started >= max_runtime - 1:
            raise RuntimeLimitExceeded(f"stopped after {max_runtime:.0f}s (the job's max_runtime_s)") from e
        raise
    ms = 1_000_000
    out = {
        "message": {"content": "".join(content), "thinking": "".join(thinking) or None},
        "done_reason": finish,
        "prompt_eval_count": timings.get("prompt_n"),
        "eval_count": timings.get("predicted_n"),
        "total_duration": int((time.monotonic() - started) * 1e9),
        "load_duration": 0,
        "prompt_eval_duration": int(timings["prompt_ms"] * ms) if "prompt_ms" in timings else None,
        "eval_duration": int(timings["predicted_ms"] * ms) if "predicted_ms" in timings else None,
    }
    if timings.get("draft_n") is not None:
        out["draft_n"], out["draft_n_accepted"] = timings.get("draft_n"), timings.get("draft_n_accepted")
    return out


def _get(url: str, timeout: float = 5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def unload_ollama(host: str, wait_s: float = 120) -> list:
    """Ask Ollama to drop every resident model and wait until it has."""
    names = [m["name"] for m in _get(f"{host}/api/ps").get("models", [])]
    for name in names:
        req = urllib.request.Request(f"{host}/api/generate", json.dumps({"model": name, "keep_alive": 0}).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60):
            pass
    deadline = time.monotonic() + wait_s
    while names and time.monotonic() < deadline:
        if not _get(f"{host}/api/ps").get("models"):
            break
        time.sleep(1)
    return names


class Manager:
    """Owns at most one llama-server process, for one model at a time."""

    def __init__(self, config: dict, ollama_host: str, pid_file: str = None, log_dir: str = None):
        self.config = config
        self.ollama_host = ollama_host
        self.port = int(config.get("port", 11438))
        self.idle_stop_s = float(config.get("idle_stop_s", 600))
        self.start_timeout_s = float(config.get("start_timeout_s", 600))
        base = os.path.dirname(BACKENDS_FILE)
        self.pid_file = pid_file or os.path.join(base, "llama-server.pid")
        self.log_dir = log_dir or base
        self.proc = None
        self.model = None
        self.ctx = 0
        self.last_used = 0.0

    def handles(self, model: str) -> bool:
        return model in (self.config.get("models") or {})

    def model_config(self, model: str) -> dict:
        return self.config["models"][model]

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def ensure(self, model: str, num_ctx: int = 0) -> float:
        """Make llama-server serve `model` with at least `num_ctx` context.
        Returns seconds spent starting it (0 if it was already up)."""
        mc = self.model_config(model)
        ctx = max(int(mc.get("ctx", 0)), int(num_ctx or 0))
        if self.running() and self.model == model and self.ctx >= ctx:
            return 0.0
        self.stop()
        if not os.path.isfile(mc["gguf"]):
            raise BackendUnavailable(f"model file missing: {mc['gguf']}")
        unload_ollama(self.ollama_host)
        t0 = time.monotonic()
        os.makedirs(self.log_dir, exist_ok=True)
        log = open(os.path.join(self.log_dir, "llama-server.log"), "ab")
        argv = [self.config["binary"], "-m", mc["gguf"], "--host", "127.0.0.1", "--port", str(self.port),
                "-c", str(ctx), "-ngl", "999", "-np", "1", "--no-webui", *mc.get("args", [])]
        try:
            self.proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        except OSError as e:
            raise BackendUnavailable(f"could not start llama-server: {e}") from e
        finally:
            log.close()
        with open(self.pid_file, "w") as f:
            f.write(str(self.proc.pid))
        while time.monotonic() - t0 < self.start_timeout_s:
            if self.proc.poll() is not None:
                self.proc = None
                raise BackendUnavailable("llama-server exited while loading (see llama-server.log)")
            try:
                if _get(f"http://127.0.0.1:{self.port}/health", timeout=2).get("status") == "ok":
                    self.model, self.ctx, self.last_used = model, ctx, time.monotonic()
                    return time.monotonic() - t0
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(1)
        self.stop()
        raise BackendUnavailable(f"llama-server not ready after {self.start_timeout_s:.0f}s")

    def touch(self) -> None:
        self.last_used = time.monotonic()

    def stop(self) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
            self.proc = None
        self.model, self.ctx = None, 0
        try:
            os.remove(self.pid_file)
        except FileNotFoundError:
            pass

    def idle_tick(self) -> bool:
        """Stop an idle server. True if it was stopped."""
        if self.running() and time.monotonic() - self.last_used > self.idle_stop_s:
            self.stop()
            return True
        return False

    def reap_stale(self) -> bool:
        """Kill a llama-server left behind by a previous worker (pid file),
        only if that pid is still a llama-server. True if one was killed."""
        try:
            pid = int(open(self.pid_file).read().strip())
        except (FileNotFoundError, ValueError):
            return False
        try:
            cmd = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout
        except OSError:
            cmd = ""
        killed = False
        if "llama-server" in cmd:
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(60):
                    try:
                        if os.waitpid(pid, os.WNOHANG)[0]:  # our own child: reaped, gone
                            break
                    except ChildProcessError:
                        pass                               # not our child (the usual case)
                    os.kill(pid, 0)
                    time.sleep(1)
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            killed = True
        try:
            os.remove(self.pid_file)
        except FileNotFoundError:
            pass
        return killed
