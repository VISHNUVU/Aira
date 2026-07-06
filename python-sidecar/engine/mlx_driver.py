"""MLX backend — macOS / Apple Silicon.

Real implementation against Apple's `mlx-lm`. This is the backend where
on-device QLoRA training actually happens (Metal GPU).

Install on the Mac:
    pip install mlx mlx-lm

mlx-lm is imported lazily so the sidecar still boots (and the rest of the app
still runs / tests) on a machine without MLX — e.g. this Linux dev box or a
Windows client. Calling an MLX method without mlx-lm installed raises a clear
EngineError telling you what to install.

Implementation status
---------------------
This is a COMPLETE implementation against the documented mlx-lm API (verified
2026), NOT a skeleton — every method is filled in. It has simply never been
executed on Metal (there's no Apple GPU in the dev sandbox). On the Mac, treat
it as verify-and-run, not write-from-scratch.

Known mlx-lm pitfalls this driver already handles (do not "simplify" these):
  * `generate()` / `stream_generate()` do NOT accept `temp=`/`temperature=`
    directly — that raises `TypeError: generate_step() got an unexpected
    keyword argument 'temp'`. You must pass `sampler=make_sampler(temp=...)`.
    This driver builds the sampler explicitly; keep it that way.
  * `stream_generate(...)` yields a response object; the text is `resp.text`
    (not the object itself).
  * `mlx_lm.lora` wants a data *directory* with train.jsonl + valid.jsonl,
    not a single file — `_prepare_data_dir()` handles the split.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Iterator, Optional

from .base import (
    EngineCapabilities,
    EngineDriver,
    EngineError,
    TrainConfig,
    TrainResult,
)

_CHANNEL_TAG_RE = re.compile(r"<\|channel>\w+\s*<channel\|>")
_CHANNEL_HOLDBACK = 32  # longest a real tag could plausibly be


def _filter_channel_tags(chunks: Iterator[str]) -> Iterator[str]:
    """Strip Gemma 4's `<|channel>NAME<channel|>` control markup from a
    stream of text chunks.

    The chat template seeds every assistant turn with a channel marker
    (observed: `<|channel>thought<channel|>`) before the model's actual
    reply. When the model never explicitly switches to a distinct channel,
    that raw markup leaks straight into what's shown to the user. This
    strips the tag wrapper (not the content after it — for short replies
    the model answers directly under the default label with no separate
    hidden reasoning to hide).

    Buffers only a small trailing window so a tag split across two streamed
    chunks still gets caught, without holding back real text waiting on
    tokens that were never going to complete a tag.
    """
    buf = ""
    for chunk in chunks:
        buf += chunk
        buf = _CHANNEL_TAG_RE.sub("", buf)
        if len(buf) > _CHANNEL_HOLDBACK:
            emit, buf = buf[:-_CHANNEL_HOLDBACK], buf[-_CHANNEL_HOLDBACK:]
            if emit:
                yield emit
    buf = _CHANNEL_TAG_RE.sub("", buf)
    if buf:
        yield buf


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _require_mlx():
    try:
        import mlx_lm  # noqa: F401
        return mlx_lm
    except ImportError as e:  # pragma: no cover - platform dependent
        raise EngineError(
            "mlx-lm is not installed. On Apple Silicon run:  pip install mlx mlx-lm\n"
            "MLX is Apple-Silicon only; on Windows/Linux use the llama.cpp driver."
        ) from e


def _is_multimodal_checkpoint(model_path: str) -> bool:
    """Detect a unified/omni checkpoint (text+vision+audio) by config.json.

    Some Gemma 4 sizes (e.g. every 12B mlx-community export) only ship as a
    unified multimodal checkpoint — `model_type: "gemma4_unified"` — which
    `mlx-lm` doesn't load (`ValueError: Model type gemma4_unified not
    supported`). `mlx-vlm` does support it, so we route to that instead.
    """
    cfg_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    model_type = str(cfg.get("model_type", ""))
    return "unified" in model_type or "vision_config" in cfg or "audio_config" in cfg


class MLXDriver(EngineDriver):
    """Gemma 4 inference + QLoRA training via mlx-lm on Metal."""

    def __init__(self, models_dir: str, adapters_dir: str,
                 embedding_model: str = "mlx-community/embeddinggemma-300m-4bit",
                 dim: int = 768):
        super().__init__(models_dir, adapters_dir)
        self._model = None
        self._tokenizer = None
        self._embedding_model_id = embedding_model
        self._embedder = None
        self._embed_tokenizer = None
        self.dim = dim
        self._vlm_mode = False
        self._vlm_config = None
        # Persistent KV caches — reused across chat turns so each request only
        # prefills the *new* part of the conversation instead of reprocessing
        # the whole history (which made latency grow with every message).
        self._vlm_cache_state = None          # mlx_vlm.PromptCacheState
        self._lm_cache = None                 # mlx_lm prompt cache (list)
        self._lm_cache_tokens: list[int] = []  # tokens the cache currently holds

    # ---- lifecycle -------------------------------------------------------
    def load(self, model_id: str, adapter_path: Optional[str] = None) -> None:
        model_path = self._resolve_model_path(model_id)
        self._vlm_mode = os.path.isdir(model_path) and _is_multimodal_checkpoint(model_path)
        self._reset_prompt_caches()

        if self._vlm_mode:
            try:
                import mlx_vlm
                from mlx_vlm.utils import load_config
            except ImportError as e:  # pragma: no cover - platform dependent
                raise EngineError(
                    "This checkpoint is a multimodal ('unified') Gemma export, which "
                    "needs mlx-vlm, not mlx-lm. Run:  pip install mlx-vlm"
                ) from e
            kw = {}
            if adapter_path:
                kw["adapter_path"] = adapter_path
            self._model, self._tokenizer = self._load_vlm_lenient(mlx_vlm, model_path, kw)
            self._vlm_config = load_config(model_path)
            self._vlm_cache_state = mlx_vlm.PromptCacheState()
        else:
            _require_mlx()
            from mlx_lm import load as mlx_load
            # mlx-lm applies a LoRA adapter at load time via adapter_path=
            kw = {}
            if adapter_path:
                kw["adapter_path"] = adapter_path
            self._model, self._tokenizer = mlx_load(model_path, **kw)

        self._model_id = model_id
        self._adapter_path = adapter_path

    @staticmethod
    def _load_vlm_lenient(mlx_vlm_module, model_path: str, kw: dict):
        """`mlx_vlm.load(..., strict=False)` doesn't actually reach the real
        `model.load_weights()` call in this installed version — the kwarg is
        silently dropped along the way, so it has no effect.

        Some Gemma 4 checkpoints (e.g. mlx-community/gemma-4-e4b-it-4bit)
        declare `num_kv_shared_layers` in config, which makes mlx_vlm skip
        building k_proj/v_proj/k_norm for the later "KV-shared" layers — but
        the checkpoint's safetensors still include those (redundant, unused)
        weights anyway, so strict loading rejects them as unknown parameters.
        The extras are genuinely harmless (the architecture doesn't use them),
        so we scope a lenient `load_weights` to just this call instead of
        waiting on an mlx_vlm fix.
        """
        import mlx.nn as nn
        original = nn.Module.load_weights

        def _lenient(self, weights, strict=True):
            return original(self, weights, strict=False)

        nn.Module.load_weights = _lenient
        try:
            return mlx_vlm_module.load(model_path, **kw)
        finally:
            nn.Module.load_weights = original

    def _reset_prompt_caches(self) -> None:
        self._vlm_cache_state = None
        self._lm_cache = None
        self._lm_cache_tokens = []

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_id = None
        self._vlm_mode = False
        self._vlm_config = None
        self._reset_prompt_caches()

    def _resolve_model_path(self, model_id: str) -> str:
        """Map a short id (e.g. 'gemma-4-e4b') to a local dir or HF repo id."""
        local = os.path.join(self.models_dir, model_id)
        if os.path.isdir(local):
            return local
        # Fall back to a HF repo id mapping (downloaded on first use by mlx-lm).
        hf_map = {
            "gemma-4-e4b": "mlx-community/gemma-4-e4b-it-4bit",
            "gemma-4-e2b": "mlx-community/gemma-4-e2b-it-4bit",
            "gemma-4-12b": "mlx-community/gemma-4-12b-it-4bit",
        }
        return hf_map.get(model_id, model_id)

    # ---- download ----------------------------------------------------------
    def download(self, model_id: str, repo_id: str, progress_cb=None) -> str:
        """Fetch weights from the Hub into ``models_dir/model_id``.

        huggingface_hub doesn't expose a stable per-byte callback across
        versions, so progress is tracked by polling actual bytes on disk
        (including partial ``*.incomplete`` files) against the repo's total
        size — the same thing `du -sh` on the target dir would show.

        Forces the classic HTTP downloader (not the optional `hf_xet`
        accelerator): xet stages chunks in its own content-addressed cache
        and only materializes complete files at the end, so disk-size
        polling would see no progress for long stretches, then a jump.
        """
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from huggingface_hub import snapshot_download, HfApi
        import threading
        import time

        local_dir = os.path.join(self.models_dir, model_id)
        os.makedirs(local_dir, exist_ok=True)

        total_bytes = 0
        try:
            info = HfApi().model_info(repo_id, files_metadata=True)
            total_bytes = sum((s.size or 0) for s in info.siblings)
        except Exception:
            pass  # progress will just show bytes downloaded, no percent

        stop = threading.Event()

        def _poll():
            while not stop.wait(0.5):
                if progress_cb:
                    progress_cb(_dir_size(local_dir), total_bytes)

        poller = threading.Thread(target=_poll, daemon=True)
        poller.start()
        try:
            path = snapshot_download(repo_id=repo_id, local_dir=local_dir)
        finally:
            stop.set()
            poller.join(timeout=2)
        if progress_cb:
            final_total = total_bytes or _dir_size(local_dir)
            progress_cb(_dir_size(local_dir), final_total)
        return path

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
        if self._model is None:
            raise EngineError("No model loaded. Call load() first.")

        if self._vlm_mode:
            yield from _filter_channel_tags(self._generate_vlm(
                messages, max_tokens=max_tokens, temperature=temperature, stream=stream))
            return

        yield from _filter_channel_tags(self._generate_lm(
            messages, max_tokens=max_tokens, temperature=temperature,
            tools=tools, stream=stream))

    def _generate_lm(self, messages, *, max_tokens, temperature, tools, stream):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        # Gemma 4 supports the `system` role and native tool specs via the
        # chat template. Passing tools= lets the template emit tool-call syntax.
        tmpl_kw = {"add_generation_prompt": True}
        if tools:
            tmpl_kw["tools"] = tools
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, **tmpl_kw
        )
        suffix_tokens = self._lm_reuse_cache(prompt)
        sampler = make_sampler(temp=temperature)
        # Always drive stream_generate internally (even for stream=False) so
        # both paths share the persistent prompt cache and per-token tracking.
        chunks: list[str] = []
        for resp in stream_generate(
            self._model, self._tokenizer, suffix_tokens,
            max_tokens=max_tokens, sampler=sampler,
            prompt_cache=self._lm_cache,
        ):
            self._lm_cache_tokens.append(resp.token)
            if stream:
                yield resp.text
            else:
                chunks.append(resp.text)
        if not stream:
            yield "".join(chunks)

    def _lm_reuse_cache(self, prompt: str) -> list[int]:
        """Trim the persistent KV cache to the shared prefix with `prompt` and
        return only the tokens that still need prefilling.

        In a chat, turn N+1's prompt starts with (almost exactly) turn N's
        prompt plus the generated reply — all of which is already in the
        cache. Without this, prefill cost grows with conversation length;
        with it, each request only pays for the newest message.
        """
        from mlx_lm.models.cache import (
            make_prompt_cache, can_trim_prompt_cache, trim_prompt_cache,
        )
        tok = self._tokenizer
        add_special = tok.bos_token is None or not prompt.startswith(tok.bos_token)
        tokens = tok.encode(prompt, add_special_tokens=add_special)

        if self._lm_cache is None:
            self._lm_cache = make_prompt_cache(self._model)
            self._lm_cache_tokens = []

        prev = self._lm_cache_tokens
        # generate_step needs at least one input token, so never match the
        # full prompt — always leave >=1 token as suffix.
        limit = min(len(prev), len(tokens) - 1)
        n = 0
        while n < limit and prev[n] == tokens[n]:
            n += 1
        if n < len(prev):
            if can_trim_prompt_cache(self._lm_cache):
                trim_prompt_cache(self._lm_cache, len(prev) - n)
            else:  # cache type can't rewind — start over
                self._lm_cache = make_prompt_cache(self._model)
                n = 0
        self._lm_cache_tokens = list(tokens)
        return tokens[n:]

    def _generate_vlm(self, messages: list[dict], *, max_tokens: int,
                      temperature: float, stream: bool) -> Iterator[str]:
        """Text-only chat through mlx-vlm, for unified/omni checkpoints.

        Native tool-call specs aren't wired through mlx-vlm's chat template
        (no `tools=` kwarg in `apply_chat_template` here) — function calling
        is a known gap for these checkpoints until mlx-vlm exposes it.

        `prompt_cache_state` is mlx_vlm's built-in cross-turn KV cache: it
        finds the shared token prefix with the previous turn itself and only
        prefills the new part (and updates itself when generation finishes).
        """
        import mlx_vlm
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt = apply_chat_template(self._tokenizer, self._vlm_config, messages)
        chunks: list[str] = []
        for resp in mlx_vlm.stream_generate(
            self._model, self._tokenizer, prompt,
            max_tokens=max_tokens, temperature=temperature,
            prompt_cache_state=self._vlm_cache_state,
        ):
            if stream:
                yield resp.text
            else:
                chunks.append(resp.text)
        if not stream:
            yield "".join(chunks)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed with a local MLX embedding model (default: EmbeddingGemma).

        mlx_embeddings' module-level `generate()` helper calls
        `model(input_ids=..., attention_mask=...)`, but EmbeddingGemma's
        `__call__` takes a positional `inputs` array instead — that mismatch
        raises `TypeError: got an unexpected keyword argument 'input_ids'`.
        So we tokenize and call the model directly. Output is
        `BaseModelOutput.text_embeds`, shape (n, dim); truncated/padded to
        `self.dim` so it matches the vector store's fixed dimensionality
        (see app.py's `embed_dim`).
        """
        if self._embedder is None:
            self._load_embedder()
        enc = self._embed_tokenizer.batch_encode_plus(
            texts, return_tensors="mlx", padding=True
        )
        out = self._embedder(enc["input_ids"], attention_mask=enc.get("attention_mask"))
        vecs = out.text_embeds if hasattr(out, "text_embeds") else out
        rows = [list(map(float, v)) for v in vecs.tolist()]
        return [self._fit_dim(v) for v in rows]

    def _fit_dim(self, vec: list[float]) -> list[float]:
        if len(vec) == self.dim:
            return vec
        if len(vec) > self.dim:
            return vec[: self.dim]
        return vec + [0.0] * (self.dim - len(vec))

    def _load_embedder(self):  # pragma: no cover - platform dependent
        try:
            from mlx_embeddings import load as load_emb
        except ImportError as e:
            raise EngineError(
                "mlx-embeddings not installed. Run: pip install mlx-embeddings"
            ) from e
        self._embedder, self._embed_tokenizer = load_emb(self._embedding_model_id)

    # ---- training --------------------------------------------------------
    def train_lora(
        self, dataset_path: str, out_dir: str, config: TrainConfig
    ) -> TrainResult:
        """Run QLoRA via the `mlx_lm.lora` CLI.

        mlx-lm expects a data *directory* containing train.jsonl / valid.jsonl.
        `dataset_path` may be that directory, or a single .jsonl we split.
        """
        if self._vlm_mode:
            return TrainResult(
                ok=False,
                error="QLoRA fine-tuning isn't wired up for unified/multimodal "
                      "checkpoints yet (mlx_lm.lora is text-only). Train on "
                      "gemma-4-e4b instead — it's the intended training target.",
            )
        _require_mlx()
        os.makedirs(out_dir, exist_ok=True)
        data_dir = self._prepare_data_dir(dataset_path, out_dir, config)
        model_path = self._resolve_model_path(self._model_id or "gemma-4-e4b")

        cmd = [
            sys.executable, "-m", "mlx_lm.lora",
            "--model", model_path,
            "--train",
            "--data", data_dir,
            "--adapter-path", out_dir,
            "--iters", str(config.iters),
            "--batch-size", str(config.batch_size),
            "--num-layers", str(config.lora_layers),
            "--learning-rate", str(config.learning_rate),
            "--max-seq-length", str(config.max_seq_len),
            "--seed", str(config.seed),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, check=False
            )
        except FileNotFoundError as e:  # pragma: no cover
            return TrainResult(ok=False, error=f"failed to launch trainer: {e}")

        if proc.returncode != 0:
            return TrainResult(
                ok=False,
                error=f"mlx_lm.lora exited {proc.returncode}: {proc.stderr[-2000:]}",
            )
        losses = self._parse_losses(proc.stdout)
        return TrainResult(
            ok=True,
            adapter_path=out_dir,
            train_loss=losses,
            meta={"stdout_tail": proc.stdout[-1000:]},
        )

    def _prepare_data_dir(self, dataset_path: str, out_dir: str,
                          config: TrainConfig) -> str:
        """Ensure an mlx-lm-style data dir (train.jsonl + valid.jsonl)."""
        if os.path.isdir(dataset_path):
            return dataset_path
        # Single JSONL -> split into train/valid inside out_dir.
        import random
        data_dir = os.path.join(out_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        with open(dataset_path) as f:
            rows = [ln for ln in f if ln.strip()]
        rng = random.Random(config.seed)
        rng.shuffle(rows)
        n_val = max(1, int(len(rows) * 0.15))
        valid, train = rows[:n_val], rows[n_val:]
        with open(os.path.join(data_dir, "train.jsonl"), "w") as f:
            f.writelines(train)
        with open(os.path.join(data_dir, "valid.jsonl"), "w") as f:
            f.writelines(valid)
        return data_dir

    @staticmethod
    def _parse_losses(stdout: str) -> list:
        """Extract (iter, train_loss) pairs from mlx-lm training output."""
        out = []
        for line in stdout.splitlines():
            if "Iter" in line and "Train loss" in line:
                try:
                    it = int(line.split("Iter")[1].split(":")[0].strip())
                    loss = float(line.split("Train loss")[1]
                                 .split(",")[0].replace(":", "").strip())
                    out.append((it, loss))
                except (ValueError, IndexError):
                    continue
        return out

    def set_adapter(self, adapter_path: Optional[str]) -> None:
        """Reload the current model with the given adapter (mlx applies at load)."""
        if self._model_id is None:
            raise EngineError("No model loaded.")
        self.load(self._model_id, adapter_path=adapter_path)

    # ---- introspection ---------------------------------------------------
    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="mlx",
            can_generate=True,
            can_embed=True,
            can_train=True,
            supports_adapters=True,
            device="metal",
            notes="Apple Silicon only. On-device QLoRA supported.",
        )
