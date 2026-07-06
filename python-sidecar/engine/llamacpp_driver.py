"""llama.cpp backend — Windows / Linux / mobile.

STUB. This is the cross-platform inference path for machines without Apple
Silicon (and the mobile clients). The interface is complete and the class is
importable/constructible everywhere so the rest of the app runs; each method
that needs the real backend raises a clear EngineError with a TODO pointer.

To implement on the target platform:
    pip install llama-cpp-python        # CPU / CUDA / Vulkan / Metal builds
Then fill in the TODOs below using GGUF-quantized Gemma 4 weights.

Training note: llama.cpp does not train. On Windows/Linux with a GPU, run the
QLoRA fine-tune via a separate HF/PEFT path (out of scope for this driver) and
load the resulting adapter here, or sync an adapter produced on the desktop.
Phones are inference-only clients.
"""
from __future__ import annotations

from typing import Iterator, Optional

from .base import (
    EngineCapabilities,
    EngineDriver,
    EngineError,
    TrainConfig,
    TrainResult,
)

_NOT_IMPL = (
    "LlamaCppDriver is a stub. Install llama-cpp-python and implement this "
    "method (see engine/llamacpp_driver.py TODOs)."
)


class LlamaCppDriver(EngineDriver):
    """GGUF inference via llama.cpp. Fill in the TODOs to activate."""

    def __init__(self, models_dir: str, adapters_dir: str, n_ctx: int = 8192,
                 n_gpu_layers: int = -1):
        super().__init__(models_dir, adapters_dir)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._llm = None

    def load(self, model_id: str, adapter_path: Optional[str] = None) -> None:
        # TODO: from llama_cpp import Llama
        #   gguf = self._resolve_gguf(model_id)
        #   self._llm = Llama(model_path=gguf, n_ctx=self.n_ctx,
        #                     n_gpu_layers=self.n_gpu_layers,
        #                     lora_path=adapter_path)  # GGUF LoRA
        #   self._model_id = model_id; self._adapter_path = adapter_path
        raise EngineError(_NOT_IMPL + " [load]")

    def unload(self) -> None:
        self._llm = None
        self._model_id = None

    def generate(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        tools: Optional[list[dict]] = None,
        stream: bool = True,
    ) -> Iterator[str]:
        # TODO: use self._llm.create_chat_completion(messages=messages,
        #   tools=tools, stream=stream, max_tokens=max_tokens,
        #   temperature=temperature) and yield delta content.
        raise EngineError(_NOT_IMPL + " [generate]")

    def embed(self, texts: list[str]) -> list[list[float]]:
        # TODO: load a GGUF embedding model with embedding=True and return
        #   self._embedder.create_embedding(texts). Alternatively route
        #   embeddings through a small ONNX model shared across platforms.
        raise EngineError(_NOT_IMPL + " [embed]")

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
        # TODO: reload with lora_path=adapter_path (llama.cpp applies LoRA at
        #   model construction; hot-swap requires reload).
        raise EngineError(_NOT_IMPL + " [set_adapter]")

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="llamacpp",
            can_generate=True,      # once implemented
            can_embed=True,         # once implemented
            can_train=False,        # llama.cpp never trains
            supports_adapters=True,
            device="cpu",           # or 'cuda'/'vulkan' depending on build
            notes="STUB. Cross-platform inference path. No on-device training.",
        )
