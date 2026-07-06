"""Fake in-memory engine — for tests and offline development.

Deterministic, dependency-free. Lets the entire app (memory, feedback, trainer,
tools, sidecar, UI) run and be tested on any machine — including this Linux dev
box — with no model weights, no GPU, no network.

- generate(): echoes a canned response derived from the last user message.
- embed(): a deterministic hash-based bag-of-words vector (stable, cheap).
- train_lora(): simulates a training run, writing a small adapter file whose
  "quality" is a deterministic function of the number of examples, so the
  eval-and-promote gate can be exercised end to end.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from typing import Iterator, Optional

from .base import (
    EngineCapabilities,
    EngineDriver,
    TrainConfig,
    TrainResult,
)

_TOKEN = re.compile(r"[a-z0-9]+")


def hash_embed(text: str, dim: int = 256) -> list[float]:
    """Deterministic hashed bag-of-words embedding, L2-normalized."""
    vec = [0.0] * dim
    for tok in _TOKEN.findall(text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
        vec[(h // dim) % dim] += 0.5   # a second bucket to reduce collisions
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class FakeDriver(EngineDriver):
    """A deterministic stand-in engine for tests and offline runs."""

    def __init__(self, models_dir: str, adapters_dir: str, dim: int = 256):
        super().__init__(models_dir, adapters_dir)
        self.dim = dim
        self._loaded = False

    def load(self, model_id: str, adapter_path: Optional[str] = None) -> None:
        self._model_id = model_id
        self._adapter_path = adapter_path
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False
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
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        adapter_tag = f" [adapter:{os.path.basename(self._adapter_path)}]" if self._adapter_path else ""
        reply = f"[fake-reply{adapter_tag}] you said: {last_user[:120]}"
        if stream:
            for word in reply.split(" "):
                yield word + " "
        else:
            yield reply

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [hash_embed(t, self.dim) for t in texts]

    def download(self, model_id: str, repo_id: str, progress_cb=None) -> str:
        """Simulate a download so the fake engine can exercise the same
        progress-bar UI as the real MLX driver, with no network or disk cost.
        Reports plausible byte counts (not just 0-100) so the UI's MB/GB
        labels look real during demos and tests."""
        local_dir = os.path.join(self.models_dir, model_id)
        os.makedirs(local_dir, exist_ok=True)
        total_bytes = 1_800_000_000  # pretend ~1.8 GB, like the smallest real model
        steps = 100
        for step in range(1, steps + 1):
            if progress_cb:
                progress_cb(int(total_bytes * step / steps), total_bytes)
            time.sleep(0.02)
        with open(os.path.join(local_dir, "FAKE_WEIGHTS"), "w") as f:
            f.write(repo_id)
        return local_dir

    def train_lora(
        self, dataset_path: str, out_dir: str, config: TrainConfig
    ) -> TrainResult:
        """Simulate training. Writes an adapter.json whose 'skill' rises with
        the number of training rows (with diminishing returns), so downstream
        eval can score it deterministically."""
        os.makedirs(out_dir, exist_ok=True)
        rows = self._count_rows(dataset_path)
        # deterministic synthetic quality in (0,1): more data -> better, saturating
        skill = 1.0 - math.exp(-rows / 50.0)
        losses = [(i, round(2.0 * math.exp(-i / 100.0) + 0.1, 4))
                  for i in range(0, config.iters + 1, max(1, config.iters // 10))]
        with open(os.path.join(out_dir, "adapters.safetensors"), "w") as f:
            f.write("FAKE-ADAPTER")   # placeholder weight file
        with open(os.path.join(out_dir, "adapter_config.json"), "w") as f:
            json.dump({"fake": True, "skill": skill, "rows": rows,
                       "rank": config.lora_rank}, f)
        return TrainResult(
            ok=True, adapter_path=out_dir, train_loss=losses,
            val_loss=losses[-1][1] if losses else None,
            meta={"skill": skill, "rows": rows},
        )

    @staticmethod
    def _count_rows(dataset_path: str) -> int:
        path = dataset_path
        if os.path.isdir(dataset_path):
            path = os.path.join(dataset_path, "train.jsonl")
        if not os.path.isfile(path):
            return 0
        with open(path) as f:
            return sum(1 for ln in f if ln.strip())

    def set_adapter(self, adapter_path: Optional[str]) -> None:
        self._adapter_path = adapter_path

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="fake",
            can_generate=True,
            can_embed=True,
            can_train=True,
            supports_adapters=True,
            device="cpu",
            notes="Deterministic test/offline engine. Not a real model.",
        )
