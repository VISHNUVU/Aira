"""Aria sidecar — the local HTTP API the Tauri UI talks to.

Two layers:
  * ``SidecarService`` wires together the engine, memory (RAG), feedback store,
    trainer + eval gate, adapter registry, and tool registry, and exposes plain
    Python methods. This layer is directly unit/integration-testable with a fake
    engine — no socket required.
  * ``make_handler`` / ``serve`` put a dependency-free ``http.server`` JSON API in
    front of the service (localhost only, CORS for the Tauri webview).

Run:  python app.py --engine mlx --port 8765
Test: import SidecarService and call methods directly (see tests/test_integration.py).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import urlparse, parse_qs

# Ensure this directory is importable regardless of how we were launched.
# When the Tauri shell (or a user) runs `python /abs/path/app.py`, Python
# normally prepends the script's dir to sys.path — but not under PYTHONSAFEPATH
# or in some frozen contexts. Doing it explicitly makes `from store import ...`
# work everywhere. (Harmless in a PyInstaller bundle, where imports are frozen.)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from store import Store
from engine import make_engine, auto_engine_name, TrainConfig
from memory import Memory, InMemoryVectorStore, LanceVectorStore
from feedback import FeedbackStore
from trainer import Trainer, AdapterRegistry
from eval_gate import EvalGate, EmbeddingSimilarityEvaluator
from tools import default_registry
from speech import SpeechManager
from stt import SttEngine
from update_checker import UpdateChecker

ARIA_HOME = os.path.expanduser(os.environ.get("ARIA_HOME", "~/.aria"))

# Single source of truth for the shipped version — keep in lockstep with
# src-tauri/tauri.conf.json's "version" (that one drives the actual .app
# bundle Info.plist; this one is what the running sidecar reports and
# compares against GitHub releases for update checks).
APP_VERSION = "0.1.0"

# HF repo mapping (documented in ARCHITECTURE.md)
MODEL_CATALOG = {
    "gemma-4-e2b": {"repo": "mlx-community/gemma-4-e2b-it-4bit", "size_gb": 1.8,
                    "role": "inference (tiny / phones)"},
    "gemma-4-e4b": {"repo": "mlx-community/gemma-4-e4b-it-4bit", "size_gb": 5.25,
                    "role": "training target (QLoRA fits 24GB) — fast chat model"},
    "gemma-4-12b": {"repo": "mlx-community/gemma-4-12b-it-4bit", "size_gb": 7.0,
                    "role": "inference (recommended, 24GB Mac)"},
}


class _MLXThreadProxy:
    """Pins every call on the wrapped object to one dedicated worker thread.

    MLX associates its Metal command stream with the OS thread that first
    touches the GPU. `ThreadingHTTPServer` hands each HTTP request to a new
    thread, so calling `engine.generate()`/`embed()`/`stt.transcribe()`/etc.
    straight from a request handler crashes with `There is no Stream(gpu, N)
    in current thread` the moment it's a different thread than whichever one
    loaded the model (e.g. the model-download background thread). Routing
    every call through a single-worker executor makes "which thread" a
    non-issue — and since there's only one GPU, serializing this way (engine
    and STT share the same executor) is the correct behavior anyway, not
    just a workaround.

    `generate()` is a generator — calling it just constructs the generator
    (no GPU work yet), but each `next()` on it does touch the GPU, so those
    also have to hop onto the worker thread one step at a time to keep
    streaming responsive instead of eagerly draining the whole generation.
    """

    def __init__(self, wrapped, executor: Optional[concurrent.futures.ThreadPoolExecutor] = None):
        self._wrapped = wrapped
        self._executor = executor or concurrent.futures.ThreadPoolExecutor(max_workers=1)

    @property
    def driver_class_name(self) -> str:
        return type(self._wrapped).__name__

    def __getattr__(self, name):
        attr = getattr(self._wrapped, name)
        if not callable(attr):
            return attr
        if name == "generate":
            def _proxied_generate(*args, **kwargs):
                gen = self._executor.submit(attr, *args, **kwargs).result()
                while True:
                    try:
                        yield self._executor.submit(next, gen).result()
                    except StopIteration:
                        return
            return _proxied_generate

        def _proxied(*args, **kwargs):
            return self._executor.submit(attr, *args, **kwargs).result()
        return _proxied


class SidecarService:
    """All app logic, wired. UI-agnostic and directly testable."""

    def __init__(self, engine_name: str = "auto", home: str = ARIA_HOME,
                 use_lance: bool = True, embed_dim: int = 768):
        self.home = home
        os.makedirs(home, exist_ok=True)
        self.models_dir = os.path.join(home, "models")
        self.adapters_dir = os.path.join(home, "adapters")
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.adapters_dir, exist_ok=True)

        # Shared by every MLX-touching object (engine + STT) — see
        # _MLXThreadProxy for why this must be a single worker thread.
        self._gpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        if engine_name == "auto":
            engine_name = auto_engine_name()
        self.engine = _MLXThreadProxy(make_engine(
            engine_name, self.models_dir, self.adapters_dir, dim=embed_dim),
            self._gpu_executor)

        self.store = Store(os.path.join(home, "aria.db"))

        # vector store: LanceDB on disk, InMemory fallback
        vstore: Any
        if use_lance:
            try:
                vstore = LanceVectorStore(os.path.join(home, "vectors"),
                                          dim=embed_dim)
            except Exception:
                vstore = InMemoryVectorStore()
        else:
            vstore = InMemoryVectorStore()

        self.memory = Memory(self.store, self.engine, vstore)
        self.feedback = FeedbackStore(self.store)
        self.registry = AdapterRegistry(self.store)
        evaluator = EmbeddingSimilarityEvaluator()
        self.trainer = Trainer(self.store, self.engine, self.feedback,
                               adapters_dir=self.adapters_dir,
                               evaluator=evaluator)
        self.tools = default_registry(self.store, self.memory)

        # voice / speech: natural offline TTS with sentence streaming + barge-in
        self.speech = SpeechManager(
            backend=os.environ.get("ARIA_VOICE", "auto"),
            voices_dir=os.path.join(home, "voices"),
        )
        # speech-to-text: offline dictation for the chat composer's mic button
        self.stt = _MLXThreadProxy(SttEngine(), self._gpu_executor)
        self.updater = UpdateChecker(APP_VERSION)
        self._loaded = False
        self._downloads: dict = {}   # model_id -> {status, downloaded_bytes, total_bytes, percent, error}
        self._update_download: dict = {}  # {status, downloaded_bytes, total_bytes, path, error}

    # ---- lifecycle -------------------------------------------------------
    def load_model(self, model_id: str = "gemma-4-12b") -> dict:
        active = self.registry.active()
        adapter_path = active["path"] if active else None
        self.engine.load(model_id, adapter_path=adapter_path)
        self._loaded = True
        return {"model": model_id, "adapter": adapter_path}

    def status(self) -> dict:
        caps = self.engine.capabilities
        active = self.registry.active()
        return {
            "engine": self.engine.driver_class_name,
            "model": self.engine.current_model,
            "active_adapter": active["id"] if active else None,
            "loaded": self._loaded,
            "capabilities": {
                "generate": caps.can_generate, "embed": caps.can_embed,
                "train": caps.can_train, "adapters": caps.supports_adapters,
                "device": caps.device,
            },
            "memory": self.memory.stats(),
            "feedback": self.feedback.stats(),
        }

    # ---- chat ------------------------------------------------------------
    # Fixed persona system prompt. Without one, the raw instruction-tuned
    # model defaults to generic "helpful assistant" tone — emoji, "How can I
    # help you today?", restating known facts unprompted. This never changes
    # between calls, so it stays at position 0 and sits inside the stable,
    # KV-cacheable prefix rather than costing anything per turn.
    PERSONA_PROMPT = (
        "You are Aria, a private AI assistant that runs entirely on the "
        "user's own Mac. Reply the way a sharp, direct friend would: plain, "
        "natural language, no emoji, no restating what you already know about "
        "them unless it's actually relevant to the question, no \"How can I "
        "help you today?\" filler, no marketing-speak. Get straight to the "
        "answer. Match the length of your reply to the question — short "
        "questions get short answers."
    )

    def _prepare_chat(self, messages: list[dict], use_memory: bool,
                      use_tools: bool) -> tuple[list[dict], str, Optional[list[dict]]]:
        """Shared RAG-grounding + tool-spec setup for chat() and chat_stream()."""
        msgs = [{"role": "system", "content": self.PERSONA_PROMPT}] + list(messages)
        context = ""
        if use_memory and msgs:
            last_user = next((m["content"] for m in reversed(msgs)
                              if m["role"] == "user"), "")
            context = self.memory.build_context(last_user)
            if context:
                # Insert right before the newest user turn, NOT at position 0.
                # Retrieved context legitimately varies turn-to-turn — if it
                # sat at the front, every turn's changed context would shift
                # token 0 and defeat the prompt-cache prefix match for the
                # *entire* conversation, not just the new message. Inserting
                # near the end keeps the stable prefix (older turns) cacheable
                # regardless of what this turn's retrieval pulls back.
                insert_at = len(msgs) - 1
                msgs = msgs[:insert_at] + [{
                    "role": "system",
                    "content": "Relevant context from the user's data:\n" + context,
                }] + msgs[insert_at:]
        tool_specs = self.tools.specs() if use_tools else None
        return msgs, context, tool_specs

    def chat(self, messages: list[dict], use_memory: bool = True,
             use_tools: bool = True, max_tokens: int = 512) -> dict:
        """Non-streaming chat with optional RAG grounding + tool specs."""
        msgs, context, tool_specs = self._prepare_chat(messages, use_memory, use_tools)
        chunks = list(self.engine.generate(msgs, max_tokens=max_tokens,
                                            tools=tool_specs, stream=False))
        text = "".join(chunks)
        return {"content": text, "used_context": bool(context)}

    def chat_stream(self, messages: list[dict], use_memory: bool = True,
                    use_tools: bool = True, max_tokens: int = 512):
        """Token-by-token chat. Yields `{"delta": "..."}` per chunk, then a
        final `{"done": true, "used_context": bool}`.

        On a multi-billion-parameter model, generation can take 10-30s+ for a
        full reply — waiting for the whole thing before showing anything (as
        the non-streaming `/chat` does) feels broken even though it's working.
        Streaming gets the first token on screen in ~1-2s instead.
        """
        msgs, context, tool_specs = self._prepare_chat(messages, use_memory, use_tools)
        for chunk in self.engine.generate(msgs, max_tokens=max_tokens,
                                          tools=tool_specs, stream=True):
            if chunk:
                yield {"delta": chunk}
        yield {"done": True, "used_context": bool(context)}

    # ---- memory ----------------------------------------------------------
    def memory_add(self, text: str, source: str = "note") -> dict:
        ids = self.memory.ingest_text(text, source=source)
        return {"chunks_added": len(ids), "ids": ids}

    def memory_list(self, limit: int = 200) -> list[dict]:
        return self.memory.list_chunks(limit=limit)

    def memory_delete(self, chunk_id: str) -> dict:
        self.memory.delete_chunk(chunk_id)
        return {"deleted": chunk_id}

    # ---- feedback / training --------------------------------------------
    def feedback_add(self, instruction: str, preferred: str,
                     signal: str = "thumbs_up") -> dict:
        ex_id = self.feedback.capture(instruction, preferred, signal)
        return {"example_id": ex_id, "pending": self.feedback.unused_count(),
                "should_train": self.feedback.should_train()}

    def train_now(self) -> dict:
        return self.trainer.run_once()

    def training_runs(self, limit: int = 50) -> list[dict]:
        return self.trainer.list_runs(limit=limit)

    def examples_list(self, limit: int = 200) -> list[dict]:
        return self.feedback.list_examples(limit=limit)

    # ---- adapters --------------------------------------------------------
    def adapters_list(self) -> list[dict]:
        return self.registry.list_all()

    def adapter_activate(self, adapter_id: str) -> dict:
        self.registry.set_active(adapter_id)
        active = self.registry.active()
        if active:
            self.engine.set_adapter(active["path"])
        return {"active": adapter_id}

    def adapter_rollback(self, adapter_id: str) -> dict:
        row = self.registry.rollback(adapter_id)
        self.engine.set_adapter(row["path"])
        return {"active": adapter_id}

    # ---- tools -----------------------------------------------------------
    def tools_list(self) -> list[dict]:
        return self.tools.list_tools()

    def tool_toggle(self, name: str, enabled: bool) -> dict:
        self.tools.set_enabled(name, enabled)
        return {"name": name, "enabled": enabled}

    def tool_calls(self, limit: int = 100) -> list[dict]:
        return self.tools.call_log(limit=limit)

    def tool_invoke(self, name: str, arguments: dict) -> dict:
        r = self.tools.dispatch(name, arguments)
        return {"ok": r.ok, "result": r.result, "error": r.error}

    # ---- models ----------------------------------------------------------
    def models_list(self) -> dict:
        downloaded = set()
        if os.path.isdir(self.models_dir):
            downloaded = {d for d in os.listdir(self.models_dir)
                          if os.path.isdir(os.path.join(self.models_dir, d))}
        return {"catalog": MODEL_CATALOG,
                "downloaded": sorted(downloaded),
                "current": self.engine.current_model}

    def model_download(self, model_id: str) -> dict:
        """Kicks off a background download and returns immediately.

        Progress (bytes/percent) is polled via ``model_download_progress()`` —
        real downloads take minutes, so the HTTP request must not block for
        the whole transfer (the UI shows a live progress bar while polling).
        """
        repo = MODEL_CATALOG.get(model_id, {}).get("repo")
        if not repo:
            return {"ok": False, "error": f"unknown model: {model_id}"}
        existing = self._downloads.get(model_id)
        if existing and existing["status"] == "downloading":
            return {"ok": True, "started": True, "already_running": True}
        if not hasattr(self.engine, "download"):
            return {"ok": False,
                    "error": "engine has no downloader (use huggingface-cli on Mac)"}

        self._downloads[model_id] = {
            "status": "downloading", "downloaded_bytes": 0,
            "total_bytes": 0, "percent": 0.0, "error": None,
        }

        def _progress(downloaded: int, total: int) -> None:
            d = self._downloads[model_id]
            d["downloaded_bytes"] = downloaded
            d["total_bytes"] = total
            d["percent"] = round(100 * downloaded / total, 1) if total else 0.0

        def _run() -> None:
            try:
                path = self.engine.download(model_id, repo, progress_cb=_progress)
                self._downloads[model_id].update(
                    status="loading", percent=100.0, path=path)
                # Load straight into memory so "download done" == "ready to
                # chat" from the UI's perspective — no separate load step the
                # onboarding flow would otherwise have no way to trigger.
                self.load_model(model_id)
                self._downloads[model_id].update(status="done")
            except Exception as e:  # pragma: no cover - needs net
                self._downloads[model_id].update(status="error", error=str(e))

        threading.Thread(target=_run, daemon=True).start()
        return {"ok": True, "started": True}

    def model_download_progress(self, model_id: str) -> dict:
        return self._downloads.get(model_id, {
            "status": "idle", "downloaded_bytes": 0, "total_bytes": 0,
            "percent": 0.0, "error": None,
        })

    # ---- voice / speech --------------------------------------------------
    def voice_status(self) -> dict:
        return self.speech.capabilities()

    def voice_set(self, voice: Optional[str] = None, rate: Optional[float] = None,
                  backend: Optional[str] = None) -> dict:
        return self.speech.set_voice(voice=voice, rate=rate, backend=backend)

    def speak(self, text: str, voice: Optional[str] = None,
              rate: Optional[float] = None) -> dict:
        """Synthesise a string to ordered audio segments (base64 WAV/AIFF).

        The UI plays segments back-to-back for gapless speech. Long text is
        automatically split into natural sentences by the SpeechManager.
        """
        segments = self.speech.speak_text(text, voice=voice, rate=rate)
        return {"segments": segments, "count": len(segments)}

    def voice_stop(self) -> dict:
        """Barge-in: stop the current spoken turn immediately."""
        return self.speech.stop()

    # ---- speech-to-text (mic button) --------------------------------------
    def stt_status(self) -> dict:
        return {"available": self.stt.available, "model": self.stt.model_id}

    def transcribe(self, audio_b64: str, mime: str = "audio/wav") -> dict:
        """Decode a base64 audio clip to a temp file and transcribe it.

        The composer's mic button always sends WAV (encoded client-side) —
        miniaudio (mlx_audio's default reader) handles that with no extra
        native dependency. Other mime types are accepted but only work if
        ffmpeg happens to be on PATH (mlx_audio shells out to it for
        webm/m4a/ogg/opus).
        """
        if not self.stt.available:
            return {"ok": False,
                    "error": "Speech-to-text needs mlx-audio (Apple Silicon only)."}
        import base64
        import tempfile
        ext = {"audio/wav": ".wav", "audio/webm": ".webm",
               "audio/mp4": ".m4a", "audio/ogg": ".ogg"}.get(mime, ".wav")
        try:
            data = base64.b64decode(audio_b64)
        except Exception as e:
            return {"ok": False, "error": f"bad audio data: {e}"}
        fd, path = tempfile.mkstemp(suffix=ext)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            text = self.stt.transcribe(path)
            return {"ok": True, "text": text}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    # ---- app updates -------------------------------------------------------
    def update_check(self) -> dict:
        return self.updater.check()

    def update_download(self, asset_url: str, asset_name: str,
                        asset_sha256: Optional[str] = None) -> dict:
        """Kicks off a background download of a release asset to ~/Downloads
        and returns immediately — same non-blocking pattern as model
        downloads. Poll via update_download_progress()."""
        if self._update_download.get("status") == "downloading":
            return {"ok": True, "started": True, "already_running": True}

        dest = os.path.join(os.path.expanduser("~/Downloads"), asset_name)
        self._update_download = {
            "status": "downloading", "downloaded_bytes": 0,
            "total_bytes": 0, "path": None, "error": None,
        }

        def _progress(downloaded: int, total: int) -> None:
            self._update_download["downloaded_bytes"] = downloaded
            self._update_download["total_bytes"] = total

        def _run() -> None:
            try:
                path = self.updater.download(asset_url, dest, expected_sha256=asset_sha256,
                                             progress_cb=_progress)
                self._update_download.update(status="done", path=path)
            except Exception as e:
                self._update_download.update(status="error", error=str(e))

        threading.Thread(target=_run, daemon=True).start()
        return {"ok": True, "started": True}

    def update_download_progress(self) -> dict:
        return self._update_download or {"status": "idle"}

    def chat_speak(self, messages: list[dict], use_memory: bool = True,
                   use_tools: bool = True, max_tokens: int = 512,
                   voice: Optional[str] = None,
                   rate: Optional[float] = None) -> dict:
        """Chat, then speak the reply. Returns text + audio segments.

        On the Mac with a real engine this streams token-by-token into the
        SpeechManager so audio starts on the first sentence. Here (and with the
        fake engine) it synthesises the completed reply — same segment format.
        """
        result = self.chat(messages, use_memory=use_memory,
                            use_tools=use_tools, max_tokens=max_tokens)
        spoken = self.speak(result["content"], voice=voice, rate=rate)
        result["speech"] = spoken
        return result


# --------------------------------------------------------------------------
# HTTP layer (stdlib, dependency-free)
# --------------------------------------------------------------------------
def _route(service: SidecarService, method: str, path: str,
           body: dict) -> tuple[int, Any]:
    """Map (method, path) -> service call. Returns (status, payload)."""
    try:
        if path == "/status":
            return 200, service.status()
        if path == "/chat" and method == "POST":
            return 200, service.chat(body.get("messages", []),
                                     use_memory=body.get("use_memory", True),
                                     use_tools=body.get("use_tools", True),
                                     max_tokens=body.get("max_tokens", 512))
        if path == "/memory" and method == "GET":
            return 200, service.memory_list(body.get("limit", 200))
        if path == "/memory" and method == "POST":
            return 200, service.memory_add(body["text"], body.get("source", "note"))
        if path == "/memory/delete" and method == "POST":
            return 200, service.memory_delete(body["chunk_id"])
        if path == "/feedback" and method == "POST":
            return 200, service.feedback_add(body["instruction"], body["preferred"],
                                             body.get("signal", "thumbs_up"))
        if path == "/train" and method == "POST":
            return 200, service.train_now()
        if path == "/training/runs":
            return 200, service.training_runs(body.get("limit", 50))
        if path == "/examples":
            return 200, service.examples_list(body.get("limit", 200))
        if path == "/adapters":
            return 200, service.adapters_list()
        if path == "/adapters/activate" and method == "POST":
            return 200, service.adapter_activate(body["adapter_id"])
        if path == "/adapters/rollback" and method == "POST":
            return 200, service.adapter_rollback(body["adapter_id"])
        if path == "/tools":
            return 200, service.tools_list()
        if path == "/tools/toggle" and method == "POST":
            return 200, service.tool_toggle(body["name"], body["enabled"])
        if path == "/tools/calls":
            return 200, service.tool_calls(body.get("limit", 100))
        if path == "/tools/invoke" and method == "POST":
            return 200, service.tool_invoke(body["name"], body.get("arguments", {}))
        if path == "/models":
            return 200, service.models_list()
        if path == "/models/download" and method == "POST":
            return 200, service.model_download(body["model_id"])
        if path == "/models/download/progress":
            return 200, service.model_download_progress(body.get("model_id", ""))
        if path == "/models/load" and method == "POST":
            return 200, service.load_model(body["model_id"])
        if path == "/voice":
            return 200, service.voice_status()
        if path == "/voice/set" and method == "POST":
            return 200, service.voice_set(voice=body.get("voice"),
                                          rate=body.get("rate"),
                                          backend=body.get("backend"))
        if path == "/speak" and method == "POST":
            return 200, service.speak(body["text"], voice=body.get("voice"),
                                      rate=body.get("rate"))
        if path == "/voice/stop" and method == "POST":
            return 200, service.voice_stop()
        if path == "/stt":
            return 200, service.stt_status()
        if path == "/transcribe" and method == "POST":
            return 200, service.transcribe(body["audio_b64"], mime=body.get("mime", "audio/wav"))
        if path == "/update/check":
            return 200, service.update_check()
        if path == "/update/download" and method == "POST":
            return 200, service.update_download(body["asset_url"], body["asset_name"],
                                                asset_sha256=body.get("asset_sha256"))
        if path == "/update/download/progress":
            return 200, service.update_download_progress()
        if path == "/chat/speak" and method == "POST":
            return 200, service.chat_speak(body.get("messages", []),
                                           use_memory=body.get("use_memory", True),
                                           use_tools=body.get("use_tools", True),
                                           max_tokens=body.get("max_tokens", 512),
                                           voice=body.get("voice"),
                                           rate=body.get("rate"))
        return 404, {"error": f"no route: {method} {path}"}
    except KeyError as e:
        return 400, {"error": f"missing field: {e}"}
    except Exception as e:
        return 500, {"error": str(e)}


def make_handler(service: SidecarService):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: Any):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0) or 0)
            if n == 0:
                return {}
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return {}

        def do_OPTIONS(self):
            self._send(204, {})

        def do_GET(self):
            parsed = urlparse(self.path)
            qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            status, payload = _route(service, "GET", parsed.path, qs)
            self._send(status, payload)

        def do_POST(self):
            path = self.path.split("?")[0]
            if path == "/chat/stream":
                self._stream_chat(self._body())
                return
            status, payload = _route(service, "POST", path, self._body())
            self._send(status, payload)

        def _stream_chat(self, body: dict):
            """NDJSON response, one `{"delta": ...}` line per token chunk,
            written and flushed as they're generated. No Content-Length (we
            don't know the final size up front) — `Connection: close` lets
            the client treat connection-close as end-of-stream, which is all
            `fetch()` + a stream reader on the frontend needs.
            """
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for event in service.chat_stream(
                    body.get("messages", []),
                    use_memory=body.get("use_memory", True),
                    use_tools=body.get("use_tools", True),
                    max_tokens=body.get("max_tokens", 512),
                ):
                    self.wfile.write((json.dumps(event) + "\n").encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                try:
                    self.wfile.write((json.dumps({"error": str(e)}) + "\n").encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def log_message(self, *a):     # quiet
            pass

    return Handler


def serve(engine: str = "auto", port: int = 8765, model: Optional[str] = None):
    service = SidecarService(engine_name=engine)
    if model:
        service.load_model(model)
    # Bind first so we know the real port (port=0 -> OS picks a free one).
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(service))
    bound_port = httpd.server_address[1]
    # Handshake line the Tauri shell parses from stdout. MUST be the first
    # thing on stdout and flushed immediately. Any library chatter that
    # printed earlier is skipped by the shell (it scans for this prefix).
    print(f"ARIA_PORT={bound_port}", flush=True)
    # Human-readable status goes to stderr so it never pollutes the handshake.
    print(f"Aria sidecar on http://127.0.0.1:{bound_port}  (engine={engine})",
          file=sys.stderr, flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Aria local sidecar")
    ap.add_argument("--engine", default="auto",
                    choices=["auto", "mlx", "llamacpp", "fake"])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default=None, help="model id to preload")
    args = ap.parse_args()
    serve(engine=args.engine, port=args.port, model=args.model)
