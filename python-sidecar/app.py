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
import base64
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator, Optional
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
from engine import make_engine, auto_engine_name, TrainConfig, ToolCallSpan
from memory import Memory, InMemoryVectorStore, LanceVectorStore, extract_auto_facts
from web_search import extract_search_query
from skills import SkillLibrary, parse_skill_draft
from chat_history import ChatHistory
from documents import extract_text as extract_document_text
from image_gen import generate as run_image_generation, is_available as image_gen_available, ImageGenError
from image_trigger import extract_image_prompt
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
APP_VERSION = "0.1.20"

# HF repo mapping (documented in ARCHITECTURE.md). Each entry is tagged with
# the engine it runs on: MLXDriver.download() pulls a whole quantized-weights
# repo (snapshot_download), while LlamaCppDriver.download() pulls one GGUF
# `filename` out of a repo that holds many alternative quantizations —
# models_list() filters this catalog down to whichever the active engine can
# actually load, so e.g. a Windows/llama.cpp user never sees an MLX-only
# entry they can't download.
MODEL_CATALOG = {
    "gemma-4-e2b": {"engine": "mlx", "repo": "mlx-community/gemma-4-e2b-it-4bit",
                    "size_gb": 1.8, "role": "inference (tiny / phones)"},
    "gemma-4-e4b": {"engine": "mlx", "repo": "mlx-community/gemma-4-e4b-it-4bit",
                    "size_gb": 5.25,
                    "role": "training target (QLoRA fits 24GB) — fast chat model"},
    "gemma-4-12b": {"engine": "mlx", "repo": "mlx-community/gemma-4-12b-it-4bit",
                    "size_gb": 7.0, "role": "inference (recommended, 24GB Mac)"},
    "gemma-4-e2b-gguf": {"engine": "llamacpp", "repo": "unsloth/gemma-4-E2B-it-GGUF",
                         "filename": "gemma-4-E2B-it-Q4_K_M.gguf", "size_gb": 1.9,
                         "role": "inference (Windows/Linux, tiny/fast)"},
    "gemma-4-e4b-gguf": {"engine": "llamacpp", "repo": "unsloth/gemma-4-E4B-it-GGUF",
                         "filename": "gemma-4-E4B-it-Q4_K_M.gguf", "size_gb": 5.5,
                         "role": "inference (Windows/Linux, recommended)"},
    "nomic-embed-text-v1.5": {"engine": "llamacpp",
                              "repo": "nomic-ai/nomic-embed-text-v1.5-GGUF",
                              "filename": "nomic-embed-text-v1.5.Q4_K_M.gguf",
                              "size_gb": 0.1,
                              "role": "embeddings (Windows/Linux, required for memory/RAG)"},
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
        self.images_dir = os.path.join(home, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.tools = default_registry(self.store, self.memory, images_dir=self.images_dir)
        self.skills = SkillLibrary(self.store)
        self.chat_history = ChatHistory(self.store)
        self._image_gen = {"status": "idle", "prompt": None, "error": None, "result": None}

        # voice / speech: natural offline TTS with sentence streaming + barge-in
        voices_dir = os.path.join(home, "voices")
        os.makedirs(voices_dir, exist_ok=True)
        self.speech = SpeechManager(
            backend=os.environ.get("ARIA_VOICE", "auto"),
            voices_dir=voices_dir,
        )
        # speech-to-text: offline dictation for the chat composer's mic button
        self.stt = _MLXThreadProxy(SttEngine(), self._gpu_executor)
        self.updater = UpdateChecker(APP_VERSION)
        self._loaded = False
        self._downloads: dict = {}   # model_id -> {status, downloaded_bytes, total_bytes, percent, error}
        self._update_download: dict = {}  # {status, downloaded_bytes, total_bytes, path, error}
        self._auto_update: dict = {"phase": "idle", "version": None, "error": None}

    # ---- lifecycle -------------------------------------------------------
    def load_model(self, model_id: str = "gemma-4-12b") -> dict:
        active = self.registry.active()
        adapter_path = active["path"] if active else None
        self.engine.load(model_id, adapter_path=adapter_path)
        self._loaded = True
        # Remembered across restarts so the user doesn't have to reopen
        # Settings -> Models and hit Load every single time they launch the
        # app, even though the weights are already sitting on disk.
        self.store.set_meta("last_model_id", model_id)
        return {"model": model_id, "adapter": adapter_path}

    def auto_load_last_model(self) -> None:
        """Best-effort reload of whatever model was active last session.
        Runs on a background thread right after the server starts listening
        — the UI is responsive immediately, showing "Starting..." / the
        first-run banner until this finishes, instead of the whole sidecar
        blocking on a multi-GB model load before it can even bind its port."""
        last_id = self.store.get_meta("last_model_id")
        if not last_id:
            return
        if last_id not in self.models_list()["downloaded"]:
            return  # weights were removed since — fall back to manual setup
        try:
            self.load_model(last_id)
        except Exception as e:
            print(f"auto-load of {last_id} failed: {e}", file=sys.stderr)

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

    # ---- persona (system prompt) -------------------------------------------
    # A custom system prompt is the "instant" lever for changing how Aria
    # behaves — takes effect on the very next message, fully reversible,
    # no GPU time. Training (see trainer.py / eval_gate.py) is the other,
    # heavier lever: permanent, gated, needs >=1 examples and a promotion
    # that beats baseline. Neither replaces the other; a custom prompt can
    # steer tone/format right now while feedback slowly accumulates toward
    # an actual retrain.
    def get_persona(self) -> dict:
        custom = self.store.get_meta("custom_persona") or ""
        return {"prompt": custom or self.PERSONA_PROMPT,
                "is_custom": bool(custom), "default": self.PERSONA_PROMPT}

    PERSONA_MAX_CHARS = 11000  # generous room for detailed instructions; still a hard cap after the corruption incident (see set_persona)

    def set_persona(self, prompt: str) -> dict:
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("prompt cannot be empty")
        if len(prompt) > self.PERSONA_MAX_CHARS:
            raise ValueError(f"prompt is {len(prompt)} characters — max is {self.PERSONA_MAX_CHARS}")
        self.store.set_meta("custom_persona", prompt)
        return self.get_persona()

    def reset_persona(self) -> dict:
        self.store.set_meta("custom_persona", "")
        return self.get_persona()

    def _auto_save_memory(self, last_user: str) -> list[str]:
        """Zero-cost heuristic fact capture — see memory.extract_auto_facts.
        Runs on every turn's latest user message; only ever stores facts that
        match a small set of high-precision self-disclosure patterns (or an
        explicit "remember ..." instruction), so it never fires on ordinary
        chat and never costs an extra model call."""
        saved = []
        for fact in extract_auto_facts(last_user):
            if self.memory.ingest_fact_if_new(fact, source="auto"):
                saved.append(fact)
        return saved

    def _search_web_if_triggered(self, last_user: str, use_tools: bool) -> Optional[dict]:
        """Deterministic, explicit-only web search — not native function
        calling (unreliable here: the VLM driver used by the recommended
        12B/e4b checkpoints never even receives tool specs). Only fires on
        an explicit "search for .../look up .../google ..." request, and
        only if the web_search tool has been switched on in Settings ->
        Tools — it's the one capability in Aria that leaves the machine,
        so it never runs silently by default."""
        tool = self.tools.get("web_search")
        if not use_tools or not tool or not tool.enabled:
            return None
        query = extract_search_query(last_user)
        if not query:
            return None
        r = self.tools.dispatch("web_search", {"query": query, "k": 5})
        if not r.ok or not r.result:
            return None
        return {"query": query, "results": r.result}

    def _prepare_chat(self, messages: list[dict], use_memory: bool,
                      use_tools: bool) -> tuple[list[dict], str, list[str], list[str], Optional[dict]]:
        """Shared RAG-grounding setup for chat() and chat_stream(). ``use_tools``
        here only gates the deterministic web-search pre-trigger below — the
        model's own ability to additionally call tools natively is a
        separate, later gate inside _run_tool_loop()."""
        msgs = [{"role": "system", "content": self.get_persona()["prompt"]}] + list(messages)
        context = ""
        auto_saved: list[str] = []
        used_skills: list[str] = []
        used_search: Optional[dict] = None
        if use_memory and msgs:
            last_user = next((m["content"] for m in reversed(msgs)
                              if m["role"] == "user"), "")
            auto_saved = self._auto_save_memory(last_user)
            matched = self.skills.match(last_user)
            used_skills = [s["name"] for s in matched]
            context = self.memory.build_context(last_user)
            used_search = self._search_web_if_triggered(last_user, use_tools)

            extra_blocks = []
            if matched:
                skill_text = "\n\n".join(
                    f'Skill "{s["name"]}": {s["instructions"]}' for s in matched)
                extra_blocks.append(
                    "The user's message matched a skill they defined. Follow "
                    "its instructions for this reply:\n" + skill_text)
            if context:
                extra_blocks.append("Relevant context from the user's data:\n" + context)
            if used_search:
                results_text = "\n".join(
                    f'- {x["title"]}: {x["snippet"]} ({x["url"]})'
                    for x in used_search["results"])
                extra_blocks.append(
                    f'Web search results for "{used_search["query"]}" — cite '
                    f"them naturally if you use them:\n{results_text}")

            if extra_blocks:
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
                    "content": "\n\n---\n\n".join(extra_blocks),
                }] + msgs[insert_at:]
        return msgs, context, auto_saved, used_skills, used_search

    # Shown when the engine's tool-call-leak filter (mlx_driver's
    # _filter_tool_call_leak) truncates a reply down to nothing — the model
    # attempted a native tool call with no preamble text before it. This
    # turns a silently-empty reply into an actionable one, since the
    # confirmed real-world trigger is exactly this: asking Aria to search
    # for something without web_search enabled.
    NO_REPLY_FALLBACK = (
        "I tried to look that up but couldn't complete it. If you want live "
        "web results, enable Web Search in Settings -> Tools and ask again "
        "with \"search for ...\" — otherwise try rephrasing so I can answer "
        "from what I already know."
    )

    def _last_user_text(self, messages: list[dict]) -> str:
        return next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")

    # ---- image generation --------------------------------------------------
    def _check_image_gen_ready(self, prompt: str) -> dict:
        """Validates a prompt against current state without starting
        anything — split out from image_generate() so chat()/chat_stream()
        can learn ok/error, save the chat message, and only then launch the
        job with that message's id already attached (see _launch_image_job).
        Launching first and attaching the id after is a real race: a fast
        or synchronously-mocked job can finish before the id is set, and
        the final result then never reaches chat_history."""
        prompt = (prompt or "").strip()
        if not prompt:
            return {"ok": False, "error": "prompt cannot be empty"}
        if self._image_gen.get("status") == "running":
            return {"ok": True, "started": True, "already_running": True,
                     "status": "running", "prompt": prompt}
        if not image_gen_available():
            return {"ok": False, "error": "image generation isn't available in this build"}
        return {"ok": True, "started": True, "prompt": prompt, "status": "running"}

    def _launch_image_job(self, prompt: str, message_id: Optional[str] = None) -> None:
        # ok=True is set once, here, and never touched again by the update()
        # calls below — it means "this job started successfully", which is
        # true for the whole lifetime of a launched job regardless of
        # whether generation later succeeds (status=done) or fails
        # (status=error). The frontend's renderImageJob() checks this field
        # first on every reload of a persisted message (see
        # chat_history.update_message_image_job) — without it, a completed
        # job with no "ok" key reads as falsy and renders a false
        # "couldn't start" error even though generation actually succeeded.
        self._image_gen = {"status": "running", "prompt": prompt, "error": None,
                            "result": None, "message_id": message_id, "ok": True}

        def _run() -> None:
            # image_gen touches MLX/Metal, so the actual generation call
            # must run on the same shared single-worker GPU thread as the
            # chat engine and STT (see _MLXThreadProxy) — this outer daemon
            # thread only exists so the HTTP handler returns immediately.
            try:
                result = self._gpu_executor.submit(
                    run_image_generation, prompt, self.images_dir).result()
                result["url"] = f"/images/file?filename={result['filename']}"
                self._image_gen.update(status="done", result=result)
            except ImageGenError as e:
                self._image_gen.update(status="error", error=str(e))
            mid = self._image_gen.get("message_id")
            if mid:
                self.chat_history.update_message_image_job(mid, dict(self._image_gen))

        threading.Thread(target=_run, daemon=True).start()

    def image_generate(self, prompt: str) -> dict:
        """Kicks off background image generation, returns immediately — poll
        via image_generate_progress(). Same non-blocking pattern as model
        downloads and training: first use can involve a ~4.3GB model
        download, and even a warm generation takes several seconds, neither
        of which should block the HTTP request thread or hang the UI."""
        job = self._check_image_gen_ready(prompt)
        if job.get("ok") and not job.get("already_running"):
            self._launch_image_job(job["prompt"])
        return job

    def image_generate_progress(self) -> dict:
        return self._image_gen

    def image_list(self, limit: int = 50) -> list[dict]:
        """Recently generated images, newest first — a lightweight gallery.
        Filenames are always our own uuid4()+".png" (see image_gen.py), so
        no path-traversal concern serving them back by filename alone."""
        if not os.path.isdir(self.images_dir):
            return []
        files = sorted(
            (f for f in os.listdir(self.images_dir) if f.endswith(".png")),
            key=lambda f: os.path.getmtime(os.path.join(self.images_dir, f)),
            reverse=True,
        )
        return [{"filename": f, "url": f"/images/file?filename={f}"} for f in files[:limit]]

    def _generate_image_if_triggered(self, last_user: str, use_tools: bool) -> Optional[dict]:
        """Deterministic, explicit-only image generation — same reasoning as
        _search_web_if_triggered: never native model tool-calling, only an
        explicit "generate/draw/create a picture of ..." phrasing, and only
        if the image_generation tool has been switched on in Settings ->
        Tools (off by default — real GPU time + a large one-time download,
        not something that should ever fire silently)."""
        tool = self.tools.get("image_generation")
        if not use_tools or not tool or not tool.enabled:
            return None
        prompt = extract_image_prompt(last_user)
        if not prompt:
            return None
        return self._check_image_gen_ready(prompt)

    def _save_image_job_turn(self, session_id: str, last_user: str, image_job: dict) -> str:
        """Persists the user message + the "generating..."/error placeholder,
        then — if this is a fresh job (not just piggybacking on one already
        running) — launches it with this message's id already attached, so
        its background thread can update chat_history with the final result
        once it finishes (see _launch_image_job / update_message_image_job).
        The job is deliberately not started until after the message exists:
        starting first and attaching the id after is a race a fast enough
        job could win, leaving the persisted message stuck on "running"."""
        text = (f'Generating "{image_job.get("prompt", last_user)}" now — '
                f"it'll appear here shortly." if image_job.get("ok")
                else image_job.get("error", "Couldn't start image generation."))
        self.chat_history.add_message(session_id, "user", last_user)
        msg = self.chat_history.add_message(session_id, "assistant", text, image_job=image_job)
        if image_job.get("ok") and not image_job.get("already_running") and msg.get("id"):
            self._launch_image_job(image_job["prompt"], message_id=msg["id"])
        return text

    # Tool specs get offered to generate() for real now (see _run_tool_loop) —
    # capped so a model that keeps calling tools can't loop forever.
    MAX_TOOL_ITERATIONS = 4

    def _run_tool_loop(self, msgs: list[dict], max_tokens: int, use_tools: bool,
                       turn_id: str, stream: bool) -> Iterator[dict]:
        """Runs generate() in a loop, letting the model natively decide to
        call any enabled tool (except image_generation — its async, polled
        result doesn't fit synchronous dispatch-and-continue, so it stays
        exclusively on its own short-circuit path in
        _generate_image_if_triggered) and see the result before continuing,
        up to MAX_TOOL_ITERATIONS rounds.

        Yields {"delta": text} chunks as they stream — first-token latency
        for the common no-tool-call reply is unaffected, since
        engine.generate() only ever buffers when it actually captures a
        tool-call span (see mlx_driver._split_tool_call) — then a final
        {"used_tools": [...]} event once the loop ends (no more tool calls,
        or the round cap was hit).
        """
        tool_specs = None
        if use_tools:
            tool_specs = [s for s in self.tools.specs(enabled_only=True)
                         if s["function"]["name"] != "image_generation"] or None
        used_tools: list[str] = []
        for _ in range(self.MAX_TOOL_ITERATIONS):
            got_tool_call = False
            for chunk in self.engine.generate(msgs, max_tokens=max_tokens,
                                              tools=tool_specs, stream=stream):
                if isinstance(chunk, ToolCallSpan):
                    calls = self.engine.parse_tool_calls(chunk.raw_text, tool_specs)
                    if not calls:
                        break  # malformed/unparseable — treat as end of turn
                    tool_calls_msg = {"role": "assistant", "content": "", "tool_calls": []}
                    tool_result_msgs = []
                    for i, call in enumerate(calls):
                        name = call.get("name", "")
                        arguments = call.get("arguments") or {}
                        call_id = f"call_{turn_id}_{len(used_tools)}_{i}"
                        tool_calls_msg["tool_calls"].append({
                            "id": call_id, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        })
                        result = self.tools.dispatch(name, arguments, turn_id=turn_id)
                        used_tools.append(name)
                        tool_result_msgs.append({
                            "role": "tool", "tool_call_id": call_id,
                            "content": json.dumps(
                                result.result if result.ok else {"error": result.error}),
                        })
                    msgs.append(tool_calls_msg)
                    msgs.extend(tool_result_msgs)
                    got_tool_call = True
                    break
                elif chunk:
                    yield {"delta": chunk}
            if not got_tool_call:
                yield {"used_tools": used_tools}
                return
        yield {"used_tools": used_tools}

    def chat(self, messages: list[dict], use_memory: bool = True,
             use_tools: bool = True, max_tokens: int = 512,
             session_id: Optional[str] = None) -> dict:
        """Non-streaming chat with optional RAG grounding + tools. Deterministic
        capabilities (memory, skills, web search pre-trigger) are injected
        server-side in _prepare_chat as before; on top of that, the model can
        now natively call any additionally-enabled tool via _run_tool_loop."""
        session_id = self.chat_history.ensure_session(session_id, self._last_user_text(messages))
        last_user = self._last_user_text(messages)
        image_job = self._generate_image_if_triggered(last_user, use_tools)
        if image_job is not None:
            # Short-circuits the normal LLM turn entirely: the "reply" here
            # is the image itself (generating in the background), not text
            # worth spending 10-30s of LLM generation on too.
            text = self._save_image_job_turn(session_id, last_user, image_job)
            return {"content": text, "used_context": False, "auto_saved": [],
                    "used_skills": [], "used_search": None,
                    "session_id": session_id, "image_job": image_job}
        msgs, context, auto_saved, used_skills, used_search = self._prepare_chat(
            messages, use_memory, use_tools)
        full_text = ""
        used_tools: list[str] = []
        for event in self._run_tool_loop(msgs, max_tokens, use_tools,
                                         turn_id=session_id, stream=False):
            if "delta" in event:
                full_text += event["delta"]
            elif "used_tools" in event:
                used_tools = event["used_tools"]
        text = full_text.strip() or self.NO_REPLY_FALLBACK
        self.chat_history.add_message(session_id, "user", last_user)
        self.chat_history.add_message(session_id, "assistant", text)
        return {"content": text, "used_context": bool(context),
                "auto_saved": auto_saved, "used_skills": used_skills,
                "used_search": used_search, "used_tools": used_tools,
                "session_id": session_id}

    def chat_stream(self, messages: list[dict], use_memory: bool = True,
                    use_tools: bool = True, max_tokens: int = 512,
                    session_id: Optional[str] = None):
        """Token-by-token chat. Yields `{"delta": "..."}` per chunk, then a
        final `{"done": true, "used_context": bool, "auto_saved": [...]}`.

        On a multi-billion-parameter model, generation can take 10-30s+ for a
        full reply — waiting for the whole thing before showing anything (as
        the non-streaming `/chat` does) feels broken even though it's working.
        Streaming gets the first token on screen in ~1-2s instead.
        """
        session_id = self.chat_history.ensure_session(session_id, self._last_user_text(messages))
        last_user = self._last_user_text(messages)
        image_job = self._generate_image_if_triggered(last_user, use_tools)
        if image_job is not None:
            text = self._save_image_job_turn(session_id, last_user, image_job)
            yield {"delta": text}
            yield {"done": True, "session_id": session_id, "used_context": False,
                  "auto_saved": [], "used_skills": [], "used_search": None,
                  "image_job": image_job}
            return
        msgs, context, auto_saved, used_skills, used_search = self._prepare_chat(
            messages, use_memory, use_tools)
        got_any = False
        full_text = ""
        used_tools: list[str] = []
        for event in self._run_tool_loop(msgs, max_tokens, use_tools,
                                         turn_id=session_id, stream=True):
            if "delta" in event:
                got_any = True
                full_text += event["delta"]
                yield {"delta": event["delta"]}
            elif "used_tools" in event:
                used_tools = event["used_tools"]
        if not got_any:
            # See NO_REPLY_FALLBACK — a tool-call-leak attempt with no
            # preamble text truncates the whole reply to nothing.
            full_text = self.NO_REPLY_FALLBACK
            yield {"delta": full_text}
        self.chat_history.add_message(session_id, "user", self._last_user_text(messages))
        self.chat_history.add_message(session_id, "assistant", full_text)
        yield {"done": True, "session_id": session_id, "used_context": bool(context),
              "auto_saved": auto_saved, "used_skills": used_skills,
              "used_search": used_search, "used_tools": used_tools}

    # ---- memory ----------------------------------------------------------
    def memory_add(self, text: str, source: str = "note") -> dict:
        ids = self.memory.ingest_text(text, source=source)
        return {"chunks_added": len(ids), "ids": ids}

    def memory_upload(self, filename: str, content_b64: str) -> dict:
        """Extracts text from an uploaded .txt/.md/.pdf and ingests it —
        the file-attachment counterpart to memory_add()'s paste-a-note box.
        Raises ValueError (UnsupportedFileType/FileTooLarge are subclasses)
        on anything the caller should surface as a clean 400, not a 500."""
        try:
            data = base64.b64decode(content_b64)
        except Exception:
            raise ValueError("content_b64 isn't valid base64")
        text = extract_document_text(filename, data)
        ids = self.memory.ingest_text(text, source=filename)
        return {"chunks_added": len(ids), "ids": ids, "filename": filename}

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
        # Filter to what the active engine can actually run — a Windows/
        # llama.cpp user shouldn't see MLX-only entries (and vice versa).
        # Non-inference-constrained engines (fake, used in tests) see the
        # full catalog, matching this repo's existing test expectations.
        engine_name = self.engine.capabilities.name
        if engine_name in ("mlx", "llamacpp"):
            catalog = {k: v for k, v in MODEL_CATALOG.items()
                      if v.get("engine") == engine_name}
        else:
            catalog = MODEL_CATALOG
        return {"catalog": catalog,
                "downloaded": sorted(downloaded),
                "current": self.engine.current_model}

    def model_download(self, model_id: str) -> dict:
        """Kicks off a background download and returns immediately.

        Progress (bytes/percent) is polled via ``model_download_progress()`` —
        real downloads take minutes, so the HTTP request must not block for
        the whole transfer (the UI shows a live progress bar while polling).
        """
        entry = MODEL_CATALOG.get(model_id, {})
        repo = entry.get("repo")
        if not repo:
            return {"ok": False, "error": f"unknown model: {model_id}"}
        filename = entry.get("filename")  # GGUF single-file (llamacpp) only
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
                if filename:
                    path = self.engine.download(model_id, repo, filename, progress_cb=_progress)
                else:
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

    def update_auto_pull(self) -> dict:
        """Check for an update and, if one exists, silently download +
        install it in the background — no ~/Downloads file, no click-through.
        Mirrors how the Claude desktop app updates: everything happens
        invisibly, and the UI only ever needs to show "Relaunch to update"
        once it's actually ready. Safe to call repeatedly (e.g. from a
        periodic poll): a version already downloading/installed/ready is
        never re-fetched."""
        result = self.updater.check()
        if not result.get("ok") or not result.get("update_available"):
            return result

        latest = result["latest_version"]
        if (self._auto_update.get("version") == latest
                and self._auto_update.get("phase") in ("downloading", "installing", "ready")):
            return result

        write_error = self.updater.check_target_writable()
        if write_error:
            self._auto_update = {"phase": "error", "version": latest, "error": write_error}
            return result

        self._auto_update = {"phase": "downloading", "version": latest, "error": None}

        def _run() -> None:
            try:
                dest_dir = os.path.join(os.path.expanduser("~"), ".aria", "updates")
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(dest_dir, result["asset_name"])
                path = self.updater.download(
                    result["asset_url"], dest, expected_sha256=result["asset_sha256"])
                self._auto_update["phase"] = "installing"
                self.updater.install_dmg(path)
                # The .dmg only ever existed as private plumbing to get the
                # new bundle onto disk — nothing for the user to find in
                # Downloads, nothing to clean up by hand.
                try:
                    os.remove(path)
                except OSError:
                    pass
                self._auto_update.update(phase="ready", error=None)
            except Exception as e:
                self._auto_update.update(phase="error", error=str(e))

        threading.Thread(target=_run, daemon=True).start()
        return result

    def update_auto_status(self) -> dict:
        return self._auto_update

    def update_relaunch(self) -> dict:
        """The new build is already installed on disk by update_auto_pull —
        this just launches it. The caller (web UI) quits this instance right
        after, the same handoff update_install used for the manual flow."""
        if self._auto_update.get("phase") != "ready":
            return {"ok": False, "error": "no update ready to relaunch into"}
        try:
            subprocess.Popen(["open", "-n", "/Applications/Aria.app"])
        except Exception as e:
            return {"ok": False, "error": f"failed to relaunch: {e}"}
        return {"ok": True}

    def update_download(self, asset_url: str, asset_name: str,
                        asset_sha256: Optional[str] = None) -> dict:
        """Kicks off a background download of a release asset to ~/Downloads
        and returns immediately — same non-blocking pattern as model
        downloads. Poll via update_download_progress()."""
        if self._update_download.get("status") == "downloading":
            return {"ok": True, "started": True, "already_running": True}

        write_error = self.updater.check_target_writable()
        if write_error:
            return {"ok": False, "error": write_error}

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

    def update_install(self) -> dict:
        """Swap /Applications/Aria.app for the downloaded build and relaunch
        it. Only ever acts on a download this same process completed and
        checksum-verified — never an arbitrary path from the request."""
        p = self._update_download
        if not p or p.get("status") != "done" or not p.get("path"):
            return {"ok": False, "error": "no completed download to install"}
        # A double-click / double-tap on "Install" would otherwise race two
        # ditto swaps over the same target — the second one always loses
        # (install_dmg's backup rename fails because the first swap already
        # moved the bundle) and can leave things in a confusing half-state.
        if self._update_download.get("installing"):
            return {"ok": True, "started": True, "already_running": True}
        self._update_download["installing"] = True
        try:
            self.updater.install_dmg(p["path"])
        except Exception as e:
            return {"ok": False, "error": str(e)}
        finally:
            self._update_download["installing"] = False
        try:
            # -n forces a genuinely new process. Without it, `open` on an
            # app that's still running (this one, until the caller quits
            # it moments from now) just refocuses the existing window
            # instead of launching one that actually runs the new binary
            # just swapped onto disk — the old process would then quit
            # with nothing having taken its place.
            subprocess.Popen(["open", "-n", "/Applications/Aria.app"])
        except Exception as e:
            return {"ok": False, "error": f"installed but failed to relaunch: {e}"}
        return {"ok": True}

    # ---- skills ------------------------------------------------------------
    def skills_list(self) -> list[dict]:
        return self.skills.list()

    def skills_add(self, name: str, instructions: str,
                   trigger: Optional[str] = None) -> dict:
        return self.skills.add(name, instructions, trigger)

    def skills_delete(self, skill_id: str) -> dict:
        self.skills.delete(skill_id)
        return {"ok": True}

    def skills_draft(self, description: str) -> dict:
        """Ask the loaded model to turn a freeform description into a
        structured skill (name, trigger, instructions) for the user to
        review and edit before saving — this is the "build it by chatting"
        entry point; nothing is saved until the user submits the form."""
        description = (description or "").strip()
        if not description:
            return {"ok": False, "error": "describe the skill first"}
        prompt = (
            "Turn this into a reusable skill for a private assistant. Reply "
            "with EXACTLY three lines and nothing else — no preamble, no "
            "markdown:\n"
            "NAME: <short title, 2-5 words>\n"
            "TRIGGER: <a short phrase the user would type to invoke this skill>\n"
            "INSTRUCTIONS: <the instructions the assistant should follow when "
            "this skill is active>\n\n"
            f"Description: {description}"
        )
        chunks = list(self.engine.generate(
            [{"role": "user", "content": prompt}], max_tokens=300,
            tools=None, stream=False))
        return parse_skill_draft("".join(chunks))

    def chat_speak(self, messages: list[dict], use_memory: bool = True,
                   use_tools: bool = True, max_tokens: int = 512,
                   voice: Optional[str] = None,
                   rate: Optional[float] = None,
                   session_id: Optional[str] = None) -> dict:
        """Chat, then speak the reply. Returns text + audio segments.

        On the Mac with a real engine this streams token-by-token into the
        SpeechManager so audio starts on the first sentence. Here (and with the
        fake engine) it synthesises the completed reply — same segment format.
        """
        result = self.chat(messages, use_memory=use_memory,
                            use_tools=use_tools, max_tokens=max_tokens,
                            session_id=session_id)
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
                                     max_tokens=body.get("max_tokens", 512),
                                     session_id=body.get("session_id"))
        if path == "/sessions" and method == "GET":
            return 200, service.chat_history.list_sessions(int(body.get("limit", 200)))
        if path == "/sessions" and method == "POST":
            return 200, service.chat_history.create_session(body.get("title"))
        if path == "/sessions/messages" and method == "GET":
            return 200, service.chat_history.get_messages(body["session_id"])
        if path == "/sessions/rename" and method == "POST":
            return 200, service.chat_history.rename_session(body["session_id"], body["title"])
        if path == "/sessions/delete" and method == "POST":
            service.chat_history.delete_session(body["session_id"])
            return 200, {"ok": True}
        if path == "/memory" and method == "GET":
            return 200, service.memory_list(body.get("limit", 200))
        if path == "/memory" and method == "POST":
            return 200, service.memory_add(body["text"], body.get("source", "note"))
        if path == "/memory/delete" and method == "POST":
            return 200, service.memory_delete(body["chunk_id"])
        if path == "/memory/upload" and method == "POST":
            try:
                return 200, service.memory_upload(body["filename"], body["content_b64"])
            except ValueError as e:
                return 400, {"error": str(e)}
        if path == "/images/generate" and method == "POST":
            return 200, service.image_generate(body["prompt"])
        if path == "/images/progress":
            return 200, service.image_generate_progress()
        if path == "/images" and method == "GET":
            return 200, service.image_list(int(body.get("limit", 50)))
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
        if path == "/update/auto" and method == "POST":
            return 200, service.update_auto_pull()
        if path == "/update/auto/status":
            return 200, service.update_auto_status()
        if path == "/update/relaunch" and method == "POST":
            return 200, service.update_relaunch()
        if path == "/update/download" and method == "POST":
            return 200, service.update_download(body["asset_url"], body["asset_name"],
                                                asset_sha256=body.get("asset_sha256"))
        if path == "/update/download/progress":
            return 200, service.update_download_progress()
        if path == "/update/install" and method == "POST":
            return 200, service.update_install()
        if path == "/skills" and method == "GET":
            return 200, service.skills_list()
        if path == "/skills" and method == "POST":
            try:
                return 200, service.skills_add(body["name"], body["instructions"],
                                               trigger=body.get("trigger"))
            except ValueError as e:
                return 400, {"error": str(e)}
        if path == "/skills/delete" and method == "POST":
            return 200, service.skills_delete(body["skill_id"])
        if path == "/skills/draft" and method == "POST":
            return 200, service.skills_draft(body.get("description", ""))
        if path == "/persona" and method == "GET":
            return 200, service.get_persona()
        if path == "/persona" and method == "POST":
            try:
                return 200, service.set_persona(body.get("prompt", ""))
            except ValueError as e:
                return 400, {"error": str(e)}
        if path == "/persona/reset" and method == "POST":
            return 200, service.reset_persona()
        if path == "/chat/speak" and method == "POST":
            return 200, service.chat_speak(body.get("messages", []),
                                           use_memory=body.get("use_memory", True),
                                           use_tools=body.get("use_tools", True),
                                           max_tokens=body.get("max_tokens", 512),
                                           voice=body.get("voice"),
                                           rate=body.get("rate"),
                                           session_id=body.get("session_id"))
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
            if parsed.path == "/images/file":
                self._serve_image_file(qs.get("filename", ""))
                return
            status, payload = _route(service, "GET", parsed.path, qs)
            self._send(status, payload)

        def _serve_image_file(self, filename: str):
            """Raw PNG bytes for a generated image — not a _route() JSON
            response, same reason /chat/stream isn't one. `os.path.basename`
            strips any directory component before joining, so a filename
            like "../../etc/passwd" resolves to just "passwd" inside
            images_dir instead of escaping it — belt-and-suspenders on top
            of every real filename always being our own uuid4()+".png"
            (image_gen.py), never anything a request could otherwise choose."""
            safe_name = os.path.basename(filename)
            path = os.path.join(service.images_dir, safe_name)
            if not safe_name.endswith(".png") or not os.path.isfile(path):
                self._send(404, {"error": "not found"})
                return
            with open(path, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

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
                    session_id=body.get("session_id"),
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
    if model:
        threading.Thread(target=service.load_model, args=(model,), daemon=True).start()
    else:
        # No explicit --model: reload whatever the user last used, if any,
        # so the app is usable without a trip to Settings on every launch.
        # Backgrounded so the port binds (and the UI loads) immediately
        # instead of waiting on a multi-GB model load first.
        threading.Thread(target=service.auto_load_last_model, daemon=True).start()
    httpd.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Aria local sidecar")
    ap.add_argument("--engine", default="auto",
                    choices=["auto", "mlx", "llamacpp", "fake"])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", default=None, help="model id to preload")
    args = ap.parse_args()
    serve(engine=args.engine, port=args.port, model=args.model)
