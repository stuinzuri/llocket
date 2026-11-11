"""The optional llama-server backend: request translation, reply mapping,
process management and the worker's routing/fallback."""
import importlib
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

FAKE = str(Path(__file__).resolve().parent / "fake-llama-server")
MSGS = [{"role": "user", "content": "hello"}]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_ollama():
    """A minimal Ollama: /api/ps lists one resident model until it is
    unloaded with keep_alive 0."""
    state = {"models": [{"name": "resident-model"}], "unloaded": []}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._json({"models": state["models"]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if req.get("keep_alive") == 0:
                state["unloaded"].append(req["model"])
                state["models"] = [m for m in state["models"] if m["name"] != req["model"]]
            self._json({"done": True})

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state["host"] = f"http://127.0.0.1:{srv.server_address[1]}"
    yield state
    srv.shutdown()


@pytest.fixture
def ls(tmp_path, monkeypatch):
    monkeypatch.setenv("LLOCKET_BACKENDS_FILE", str(tmp_path / "backends.json"))
    import llama_server
    importlib.reload(llama_server)
    return llama_server


def config(tmp_path, port, binary=FAKE, **model):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"")
    return {"binary": binary, "port": port, "idle_stop_s": 600, "start_timeout_s": 20,
            "models": {"my-model": {"gguf": str(gguf), "ctx": 4096, "args": ["--extra", "1"], **model}}}


def test_sampler_precedence_and_option_mapping(ls):
    body, ignored = ls.build_request(MSGS, True, {"temperature": 0.3, "num_ctx": 8192, "num_predict": 100,
                                                  "seed": 7, "made_up": 1},
                                     defaults={"temperature": 1.0, "top_k": 20})
    assert body["temperature"] == 0.3            # job beats model default
    assert body["top_k"] == 20                   # model default beats built-in
    assert body["top_p"] == ls.OLLAMA_DEFAULTS["top_p"]   # built-in always sent
    assert body["min_p"] == 0.0
    assert body["seed"] == 7 and body["max_tokens"] == 100
    assert "num_ctx" not in body and "num_predict" not in body
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert ignored == ["made_up"]
    body, _ = ls.build_request(MSGS, False, {"num_predict": -1})
    assert "max_tokens" not in body and body["chat_template_kwargs"] == {"enable_thinking": False}


def test_images_stay_on_ollama(ls):
    assert ls.supports(MSGS)
    assert not ls.supports([{"role": "user", "content": "x", "images": ["b64"]}])


def test_manager_starts_calls_and_stops(ls, tmp_path, fake_ollama, monkeypatch):
    rec = tmp_path / "args.jsonl"
    monkeypatch.setenv("FAKE_LLAMA_RECORD", str(rec))
    m = ls.Manager(config(tmp_path, free_port()), fake_ollama["host"], log_dir=str(tmp_path))
    assert m.handles("my-model") and not m.handles("other")
    assert m.ensure("my-model") > 0
    assert fake_ollama["unloaded"] == ["resident-model"]       # Ollama freed first
    assert m.ensure("my-model", 2048) == 0                     # already big enough: no restart
    args = json.loads(rec.read_text().splitlines()[0])
    assert args[args.index("-c") + 1] == "4096" and args[-2:] == ["--extra", "1"]

    body, _ = ls.build_request(MSGS, True, {"num_predict": 5})
    out = ls.call(m.port, body)
    assert out["message"]["thinking"] == "hmm"
    assert out["message"]["content"].startswith("echo:")
    sent = json.loads(out["message"]["content"][5:])
    assert sent["max_tokens"] == 5 and sent["stream"] is True
    assert out["done_reason"] == "stop"
    assert (out["prompt_eval_count"], out["eval_count"]) == (12, 7)
    assert out["prompt_eval_duration"] == 50_000_000 and out["eval_duration"] == 70_000_000
    assert (out["draft_n"], out["draft_n_accepted"]) == (6, 4)

    assert m.ensure("my-model", 16384) > 0                     # bigger context: restarted
    assert len(rec.read_text().splitlines()) == 2
    m.last_used -= 601
    assert m.idle_tick() and not m.running()
    assert not Path(m.pid_file).exists()


def test_reap_stale_only_kills_llama_server(ls, tmp_path, fake_ollama):
    m = ls.Manager(config(tmp_path, free_port()), fake_ollama["host"], log_dir=str(tmp_path))
    m.ensure("my-model")
    proc, m.proc = m.proc, None                                # a worker that died, leaving its server
    fresh = ls.Manager(m.config, fake_ollama["host"], pid_file=m.pid_file, log_dir=str(tmp_path))
    assert fresh.reap_stale()
    assert proc.poll() is not None or proc.wait(timeout=10) is not None
    Path(m.pid_file).write_text("1")                           # pid 1 is not a llama-server
    assert not fresh.reap_stale() and not Path(m.pid_file).exists()


def test_unstartable_server_raises(ls, tmp_path, fake_ollama):
    m = ls.Manager(config(tmp_path, free_port(), binary=str(tmp_path / "missing")), fake_ollama["host"],
                   log_dir=str(tmp_path))
    with pytest.raises(ls.BackendUnavailable):
        m.ensure("my-model")


def _route(q, ls, tmp_path, monkeypatch, cfg, model="my-model"):
    lib, worker = q
    (tmp_path / "backends.json").write_text(json.dumps({"llama_server": cfg}))
    importlib.reload(worker)                                    # picks up the reloaded llama_server
    calls = []
    monkeypatch.setattr(worker, "call_ollama", lambda *a, **k: calls.append((a, k)) or
                        {"message": {"content": "from ollama"}, "done_reason": "stop"})
    job = {"id": "j1", "host": worker.OLLAMA_HOST, "model": model}
    result = worker.run_model(job, MSGS, {"think": False, "sampling_options": {"num_ctx": 2048}}, None)
    return worker, result, calls


def test_worker_routes_listed_model_to_llama_server(q, ls, tmp_path, fake_ollama, monkeypatch):
    monkeypatch.setattr(q[0], "OLLAMA_HOST", fake_ollama["host"])
    monkeypatch.setenv("LLOCKET_OLLAMA_HOST", fake_ollama["host"])
    worker, result, calls = _route(q, ls, tmp_path, monkeypatch, config(tmp_path, free_port()))
    try:
        assert not calls and result["backend"] == "llama-server"
        doc = worker.output_doc(result)
        assert doc["backend"] == "llama-server" and doc["draft_n_accepted"] == 4 and doc["eval_count"] == 7
        # an unlisted model goes to Ollama, and the server is stopped first to free memory
        result = worker.run_model({"id": "j2", "host": worker.OLLAMA_HOST, "model": "other"}, MSGS, {}, None)
        assert result["backend"] == "ollama" and len(calls) == 1 and not worker.manager().running()
    finally:
        worker.manager().stop()


def test_worker_falls_back_to_ollama(q, ls, tmp_path, fake_ollama, monkeypatch):
    monkeypatch.setenv("LLOCKET_OLLAMA_HOST", fake_ollama["host"])
    cfg = config(tmp_path, free_port(), binary=str(tmp_path / "missing"))
    worker, result, calls = _route(q, ls, tmp_path, monkeypatch, cfg)
    assert result["backend"] == "ollama" and len(calls) == 1
    assert calls[0][1]["options"] == {"num_ctx": 2048}


def test_no_backends_file_means_ollama_only(q, ls, tmp_path, monkeypatch):
    lib, worker = q
    importlib.reload(worker)
    monkeypatch.setattr(worker, "call_ollama", lambda *a, **k: {"message": {"content": "x"}})
    assert worker.manager() is None
    result = worker.run_model({"id": "j", "host": worker.OLLAMA_HOST, "model": "my-model"}, MSGS, {}, None)
    assert result["backend"] == "ollama" and "backend" in worker.output_doc(result)
