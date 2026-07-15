"""llama.cpp backend — Windows / Linux / mobile.

Cross-platform inference path for machines without Apple Silicon, via
``llama-cpp-python`` over GGUF-quantized weights. Runs on CPU everywhere
(and CUDA/Metal/Vulkan if the installed wheel was built with that backend),
which is also why this driver is fully testable on a Mac dev machine even
though its real target is Windows — see ``tests/test_llamacpp_driver.py``.

Model layout mirrors MLXDriver's convention: ``models_dir/<model_id>/`` holds
the weights, here a single ``*.gguf`` file (or a path directly to one, for
tests) rather than a whole snapshot directory.
"""
from __future__ import annotations

import json
import os
from typing import Iterator, Optional

from .base import (
    EngineCapabilities,
    EngineDriver,
    EngineError,
    ToolCallSpan,
    TrainConfig,
    TrainResult,
)


class LlamaCppDriver(EngineDriver):
    """GGUF inference via llama.cpp."""

    def __init__(self, models_dir: str, adapters_dir: str, n_ctx: int = 8192,
                 n_gpu_layers: int = -1, embedding_model: str = "nomic-embed-text-v1.5",
                 dim: int = 768):
        super().__init__(models_dir, adapters_dir)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._embedding_model_id = embedding_model
        self.dim = dim
        self._llm = None
        self._embedder = None

    # ---- download ----------------------------------------------------------
    def download(self, model_id: str, repo_id: str, filename: str,
                 progress_cb=None) -> str:
        """Fetch one GGUF file from the Hub into ``models_dir/model_id/``.

        Unlike MLXDriver.download's whole-repo ``snapshot_download`` (an MLX
        checkpoint is many small shard files), a GGUF repo holds many
        *alternative quantizations* of the same model — only one ``filename``
        is wanted, hence ``hf_hub_download`` instead. Progress is tracked the
        same way (poll the file's growing size on disk against the repo's
        reported size for that one file) for the same reason: neither
        download function exposes a stable per-byte callback across
        huggingface_hub versions.
        """
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        try:
            from huggingface_hub import hf_hub_download, HfApi
        except ImportError:
            raise EngineError(
                "huggingface_hub isn't available in this build — model "
                "downloads need it even though inference itself doesn't."
            )
        import threading

        local_dir = os.path.join(self.models_dir, model_id)
        os.makedirs(local_dir, exist_ok=True)
        target_path = os.path.join(local_dir, filename)

        total_bytes = 0
        try:
            info = HfApi().model_info(repo_id, files_metadata=True)
            match = next((s for s in info.siblings if s.rfilename == filename), None)
            total_bytes = (match.size or 0) if match else 0
        except Exception:
            pass  # progress will just show bytes downloaded, no percent

        stop = threading.Event()

        def _poll():
            while not stop.wait(0.5):
                if progress_cb:
                    size = os.path.getsize(target_path) if os.path.exists(target_path) else 0
                    progress_cb(size, total_bytes)

        poller = threading.Thread(target=_poll, daemon=True)
        poller.start()
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        finally:
            stop.set()
            poller.join(timeout=2)
        if progress_cb:
            final_size = os.path.getsize(path) if os.path.exists(path) else 0
            progress_cb(final_size, total_bytes or final_size)
        return path

    # ---- lifecycle -------------------------------------------------------
    def _resolve_gguf_path(self, model_id: str) -> str:
        """``models_dir/model_id`` holding one ``*.gguf`` file, or (mainly
        for tests) ``model_id`` itself already pointing at a ``.gguf`` file."""
        if os.path.isfile(model_id) and model_id.endswith(".gguf"):
            return model_id
        model_dir = os.path.join(self.models_dir, model_id)
        if not os.path.isdir(model_dir):
            raise EngineError(f"model not found: {model_id} (expected {model_dir})")
        ggufs = sorted(f for f in os.listdir(model_dir) if f.endswith(".gguf"))
        if not ggufs:
            raise EngineError(f"no .gguf file found in {model_dir}")
        return os.path.join(model_dir, ggufs[0])

    def load(self, model_id: str, adapter_path: Optional[str] = None) -> None:
        from llama_cpp import Llama

        gguf_path = self._resolve_gguf_path(model_id)
        # "chatml-function-calling" is llama-cpp-python's model-agnostic tool
        # calling handler (grammar-constrained JSON) — used unconditionally,
        # not just when tools are actually passed, because chat_format is
        # fixed at construction time (no per-call override) and this handler
        # degrades gracefully to a plain chat reply when no tools are given.
        # Trade-off: slightly less prompt fidelity than the model's own
        # native chat template (e.g. MLXDriver applies Gemma 4's own
        # template) in exchange for uniform tool-calling support here.
        self._llm = Llama(
            model_path=gguf_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            lora_path=adapter_path,
            chat_format="chatml-function-calling",
            verbose=False,
        )
        self._model_id = model_id
        self._adapter_path = adapter_path

    def unload(self) -> None:
        self._llm = None
        self._model_id = None
        self._adapter_path = None

    # ---- inference -------------------------------------------------------
    def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        tools: Optional[list[dict]] = None,
        stream: bool = True,
    ) -> Iterator[str]:
        if self._llm is None:
            raise EngineError("no model loaded")

        if tools:
            # The grammar-constrained tool-calling path only produces a
            # complete, parseable result once generation finishes — there's
            # no meaningful partial JSON to stream — so this buffers once,
            # then mirrors MLXDriver.generate()'s shape: any plain text
            # first, then a single ToolCallSpan if the model called a tool.
            resp = self._llm.create_chat_completion(
                messages=messages, tools=tools, tool_choice="auto",
                max_tokens=max_tokens, temperature=temperature, stream=False,
            )
            msg = resp["choices"][0]["message"]
            content = msg.get("content") or ""
            if content:
                yield content
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                yield ToolCallSpan(raw_text=json.dumps(tool_calls))
            return

        if stream:
            for chunk in self._llm.create_chat_completion(
                messages=messages, max_tokens=max_tokens,
                temperature=temperature, stream=True,
            ):
                delta = chunk["choices"][0].get("delta", {})
                text = delta.get("content")
                if text:
                    yield text
        else:
            resp = self._llm.create_chat_completion(
                messages=messages, max_tokens=max_tokens,
                temperature=temperature, stream=False,
            )
            content = resp["choices"][0]["message"].get("content") or ""
            if content:
                yield content

    def parse_tool_calls(
        self, text: str, tools: Optional[list[dict]] = None
    ) -> list[dict]:
        """Parse a ``ToolCallSpan.raw_text`` captured above. Unlike MLX's
        Gemma leak-parsing, llama-cpp-python already hands back structured
        OpenAI-shaped tool calls, so ``raw_text`` here is just that list
        JSON-encoded — this only has to decode it and flatten each call's
        stringified ``arguments`` into a real dict."""
        try:
            calls = json.loads(text)
        except (TypeError, ValueError):
            return []
        if not isinstance(calls, list):
            return []
        result = []
        for call in calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except (TypeError, ValueError):
                args = {}
            result.append({"name": name, "arguments": args})
        return result

    def _load_embedder(self) -> None:
        from llama_cpp import Llama

        gguf_path = self._resolve_gguf_path(self._embedding_model_id)
        self._embedder = Llama(model_path=gguf_path, embedding=True, verbose=False)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed via a separate embedding-specific GGUF (loaded lazily,
        distinct from the chat model — llama.cpp doesn't support running
        both causal generation and embedding pooling on one instance)."""
        if self._embedder is None:
            self._load_embedder()
        resp = self._embedder.create_embedding(texts)
        rows = [d["embedding"] for d in resp["data"]]
        # A single-vector-per-input pooled embedding comes back as one flat
        # list of floats; some GGUF configs instead return per-token vectors
        # (list of lists) — collapse those with a mean pool before fitting.
        flat_rows = [r[0] if (r and isinstance(r[0], list)) else r for r in rows]
        return [self._fit_dim([float(x) for x in row]) for row in flat_rows]

    def _fit_dim(self, vec: list[float]) -> list[float]:
        if len(vec) == self.dim:
            return vec
        if len(vec) > self.dim:
            return vec[: self.dim]
        return vec + [0.0] * (self.dim - len(vec))

    # ---- training / adapters --------------------------------------------
    def train_lora(
        self, dataset_path: str, out_dir: str, config: TrainConfig
    ) -> TrainResult:
        # llama.cpp does not train. Return an informative failure so the
        # trainer/UI can degrade gracefully on this platform.
        return TrainResult(
            ok=False,
            error="llama.cpp backend cannot train. Train on the desktop "
                  "(MLX on Mac, or HF/PEFT on a CUDA box) and sync the adapter.",
        )

    def set_adapter(self, adapter_path: Optional[str]) -> None:
        # llama.cpp applies LoRA at construction time only — hot-swapping
        # means reloading the model with the new adapter path.
        if self._model_id is None:
            raise EngineError("no model loaded")
        self.load(self._model_id, adapter_path=adapter_path)

    # ---- introspection ---------------------------------------------------
    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="llamacpp",
            can_generate=True,
            can_embed=True,
            can_train=False,
            supports_adapters=True,
            device="cpu",
            notes="Cross-platform inference via llama.cpp. No on-device training.",
            # chat_format is pinned to "chatml-function-calling" at load()
            # time unconditionally (see load()'s docstring) — once a model
            # is loaded, tool-calling always works regardless of checkpoint,
            # unlike MLXDriver where it depends on the specific chat template.
            supports_tool_calling=self._llm is not None,
        )
