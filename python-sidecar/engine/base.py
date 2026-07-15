"""Pluggable engine layer.

Everything above this interface (memory, feedback, trainer, tools, UI) is
platform-independent. Only concrete drivers know about MLX vs llama.cpp.

Add a new backend by subclassing ``EngineDriver`` and registering it in
``engine/__init__.py::make_engine``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional


@dataclass
class EngineCapabilities:
    """What a given backend can do on the current platform.

    The UI reads these to hide/disable panels (e.g. no training panel on a
    backend where ``can_train`` is False).
    """
    name: str
    can_generate: bool = True
    can_embed: bool = False
    can_train: bool = False
    supports_adapters: bool = False
    device: str = "cpu"           # 'metal' | 'cuda' | 'cpu' | 'vulkan'
    notes: str = ""
    # Whether the *currently loaded* model can natively call tools — not a
    # fixed backend property like the fields above, since it depends on
    # whether this specific checkpoint's chat template declares tool-call
    # syntax (see MLXDriver._resolve_tool_calling's docstring for why this
    # isn't always knowable from the backend alone: mlx_lm.load() populates
    # this automatically, mlx_vlm.load() never did, even for an identical
    # chat template — a real, previously-invisible gap this field exists to
    # surface instead of leaving tools silently never firing with no signal
    # anywhere the UI could show).
    supports_tool_calling: bool = False


@dataclass
class TrainConfig:
    """QLoRA hyper-parameters. Sensible defaults for Gemma 4 E4B on 24 GB."""
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 1e-4
    iters: int = 600
    batch_size: int = 1
    max_seq_len: int = 4096
    lora_layers: int = 16
    seed: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class TrainResult:
    """Outcome of a fine-tune run."""
    ok: bool
    adapter_path: Optional[str] = None
    train_loss: list = field(default_factory=list)   # [(step, loss), ...]
    val_loss: Optional[float] = None
    error: Optional[str] = None
    meta: dict = field(default_factory=dict)


@dataclass
class ToolCallSpan:
    """A captured, unparsed native tool-call span from a generation stream.

    ``generate()`` yields one of these (instead of a plain str chunk) when it
    detects a complete native tool-call attempt while ``tools`` was passed —
    see MLXDriver's module docstring for why this replaces truncation. Pass
    ``raw_text`` to ``parse_tool_calls()`` to get structured calls back.
    """
    raw_text: str


class EngineError(RuntimeError):
    """Raised when a backend operation fails or is unsupported."""


class EngineDriver(ABC):
    """Abstract inference/training backend.

    Concrete implementations: :class:`MLXDriver` (macOS / Apple Silicon),
    :class:`LlamaCppDriver` (Windows / Linux, via llama.cpp).

    Tool-calling contract for a new backend
    ----------------------------------------
    ``app.py``'s ``_run_tool_loop`` drives every backend identically — it
    never branches on which driver is loaded — by relying on three things
    every driver must get right. This is deliberately a *contract*, not a
    shared base-class implementation: :class:`MLXDriver` and
    :class:`LlamaCppDriver` satisfy it via genuinely different mechanics
    (streaming tag-capture off raw text vs. a library that already returns
    structured tool calls), because their underlying libraries expose
    fundamentally different APIs — forcing one shared implementation would
    add indirection without removing real duplication. A third backend
    should satisfy the same three points however fits its own library best:

    1. ``generate()`` yields a :class:`ToolCallSpan` (instead of a plain
       ``str`` chunk) when — and only when — ``tools`` was passed *and* the
       model actually produced a complete tool-call attempt. Any plain text
       before that point should still stream through normally as ``str``
       chunks (this is what keeps first-token latency unaffected for the
       common no-tool-call reply — see :class:`ToolCallSpan`'s docstring).
    2. ``parse_tool_calls(raw_text, tools)`` turns that captured span into
       ``[{"name": str, "arguments": dict}, ...]`` — ``[]`` if the backend or
       currently-loaded checkpoint doesn't support tool-calling at all, never
       a raised exception for "no calls found" (normalize any such
       exceptions from an underlying parser library into an empty list).
    3. ``capabilities.supports_tool_calling`` accurately reflects whether the
       *currently loaded model* — not just the backend in the abstract —
       supports native tool-calling. This is model-specific for
       :class:`MLXDriver` (depends on the loaded checkpoint's chat template)
       but a fixed ``True`` once any model is loaded for
       :class:`LlamaCppDriver` (its ``chatml-function-calling`` handler
       supports tools regardless of checkpoint). A real, previously-invisible
       gap existed here: nothing surfaced whether tools would actually work
       until this field was added — see ``EngineCapabilities.supports_tool_calling``.
    """

    def __init__(self, models_dir: str, adapters_dir: str):
        self.models_dir = models_dir
        self.adapters_dir = adapters_dir
        self._model_id: Optional[str] = None
        self._adapter_path: Optional[str] = None

    # ---- lifecycle -------------------------------------------------------
    @abstractmethod
    def load(self, model_id: str, adapter_path: Optional[str] = None) -> None:
        """Load a model (and optionally an adapter) into memory."""

    @abstractmethod
    def unload(self) -> None:
        """Free model memory."""

    # ---- inference -------------------------------------------------------
    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        tools: Optional[list[dict]] = None,
        stream: bool = True,
    ) -> Iterator[str]:
        """Chat completion. Yields text chunks (or one chunk if ``stream=False``).

        ``messages`` is a list of ``{"role","content"}`` dicts. ``tools`` is a
        list of JSON-schema tool specs for native function calling.
        """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""

    def parse_tool_calls(
        self, text: str, tools: Optional[list[dict]] = None
    ) -> list[dict]:
        """Parse a captured ``ToolCallSpan.raw_text`` into structured calls:
        ``[{"name": str, "arguments": dict}, ...]``, or ``[]`` if the text
        doesn't contain a valid call or this backend/checkpoint doesn't
        support native tool-calling. Optional capability — default no-op so
        drivers without a tool-call parser (FakeDriver, LlamaCppDriver,
        MLXDriver in VLM mode) don't have to implement it."""
        return []

    # ---- training / adapters --------------------------------------------
    @abstractmethod
    def train_lora(
        self, dataset_path: str, out_dir: str, config: TrainConfig
    ) -> TrainResult:
        """Run a QLoRA fine-tune, writing an adapter to ``out_dir``."""

    def list_adapters(self) -> list[str]:
        """Adapter directories currently on disk (default: filesystem scan)."""
        import os
        if not os.path.isdir(self.adapters_dir):
            return []
        return sorted(
            os.path.join(self.adapters_dir, d)
            for d in os.listdir(self.adapters_dir)
            if os.path.isdir(os.path.join(self.adapters_dir, d))
        )

    @abstractmethod
    def set_adapter(self, adapter_path: Optional[str]) -> None:
        """Activate an adapter (or ``None`` for the base model)."""

    # ---- introspection ---------------------------------------------------
    @property
    @abstractmethod
    def capabilities(self) -> EngineCapabilities:
        ...

    @property
    def current_model(self) -> Optional[str]:
        return self._model_id

    @property
    def current_adapter(self) -> Optional[str]:
        return self._adapter_path
