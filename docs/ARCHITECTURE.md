# Architecture — Local Self-Improving AI ("Aria")

> A cross-platform, offline-first personal AI. Runs an open-weight Gemma 4 model
> locally, remembers everything you give it (RAG), and periodically retrains a
> personal LoRA adapter on your validated feedback — keeping only versions that
> measurably improve. Everything stays on your device.

---

## 1. Design goals

| Goal | How it's met |
|------|--------------|
| **Offline** | Model weights, embeddings, vector DB, and training all run locally. No network calls at inference time. |
| **Personalized** | Two mechanisms: instant **memory (RAG)** + periodic **QLoRA fine-tuning** on your feedback. |
| **Self-improving, safely** | Batched retraining behind an **eval-and-promote gate** — a new adapter is kept only if it beats the current one on a held-out set. Never degrades. |
| **Transparent** | An inspector UI exposes *everything*: memory chunks, tool calls, training runs, adapter versions, model status. |
| **Cross-platform** | One Tauri shell (Windows/macOS/Linux/iOS/Android) + a **pluggable engine layer** so the ML backend swaps per platform. |

---

## 2. Component diagram

```
┌───────────────────────────────────────────────────────────────┐
│                         TAURI SHELL                             │
│  (Rust core + web UI — Win / macOS / Linux / iOS / Android)     │
│                                                                 │
│   ┌─────────────┐  ┌──────────┐  ┌───────────┐  ┌───────────┐   │
│   │    Chat     │  │  Memory  │  │   Tools   │  │ Training / │   │
│   │             │  │ browser  │  │  + log    │  │ Adapters  │   │
│   └─────────────┘  └──────────┘  └───────────┘  └───────────┘   │
│          │  Inspector UI (HTML/JS in webview)                   │
└──────────┼──────────────────────────────────────────────────── ┘
           │  JSON-RPC over local socket (127.0.0.1 / stdio)
┌──────────┼──────────────────────────────────────────────────────┐
│          ▼                PYTHON SIDECAR                          │
│                                                                  │
│   ┌────────────────────────────────────────────────────────┐    │
│   │                    Sidecar API (app.py)                  │    │
│   │  chat · ingest · list_memory · list_tools · run_tool ·  │    │
│   │  list_adapters · trigger_train · model_status · ...      │    │
│   └───┬─────────┬──────────┬──────────┬───────────┬─────────┘    │
│       │         │          │          │           │              │
│   ┌───▼───┐ ┌───▼────┐ ┌───▼────┐ ┌───▼─────┐ ┌───▼──────┐       │
│   │Memory │ │Feedback│ │ Trainer│ │  Tools  │ │  Engine  │       │
│   │ (RAG) │ │ store  │ │ + gate │ │ registry│ │  layer   │       │
│   └───┬───┘ └───┬────┘ └───┬────┘ └─────────┘ └───┬──────┘       │
│       │         │          │                      │              │
│   ┌───▼─────────▼──────────▼───┐         ┌────────▼─────────┐    │
│   │   SQLite + vector store    │         │  EngineDriver    │    │
│   │  memory · examples ·       │         │  (abstract)      │    │
│   │  adapters · tool_calls     │         ├──────────────────┤    │
│   └────────────────────────────┘         │ MLXDriver (Mac)  │    │
│                                           │ LlamaCppDriver   │    │
│                                           │  (Win/Linux/     │    │
│                                           │   mobile stub)   │    │
│                                           └────────┬─────────┘    │
└────────────────────────────────────────────────────┼────────────┘
                                                      │
                                            ┌─────────▼─────────┐
                                            │  Gemma 4 weights  │
                                            │  + LoRA adapters  │
                                            │  (local disk)     │
                                            └───────────────────┘
```

---

## 3. Process model

Two processes, one machine:

1. **Tauri shell** — owns the window and the web UI. Written in Rust; UI in
   HTML/JS. On launch it **spawns the Python sidecar** as a child process and
   connects over a local JSON-RPC channel.
2. **Python sidecar** — owns all ML and data: the engine layer, memory, feedback,
   trainer, tools, and the databases. This is where every non-trivial line of
   logic lives, which is why it's platform-independent Python and fully unit-testable
   without the GUI.

**Why a sidecar and not pure Rust/Swift?** The ML ecosystem (mlx-lm, embeddings,
vector DBs, PyTorch/HF for training) is Python. Keeping it in a sidecar means the
same tested code runs identically under a Tauri shell, a SwiftUI shell, or a
headless CLI. The shell is a thin client.

### Transport
JSON-RPC 2.0. Default transport is a TCP socket on `127.0.0.1:<ephemeral>`
(the sidecar prints its port on startup; the shell reads it). `stdio` framing is
supported as a fallback for sandboxed/mobile contexts where opening a local port
is restricted.

---

## 4. The pluggable engine layer

The single most important abstraction for cross-platform reach. Everything above
the engine (memory, feedback, trainer orchestration, tools, UI) is
platform-independent; only the engine knows about MLX vs llama.cpp.

```python
class EngineDriver(ABC):
    def load(self, model_id: str, adapter_path: str | None = None) -> None: ...
    def generate(self, messages: list[dict], **kw) -> Iterator[str]: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def train_lora(self, dataset_path, out_dir, config) -> TrainResult: ...
    def list_adapters(self) -> list[str]: ...
    def set_adapter(self, path: str | None) -> None: ...
    @property
    def capabilities(self) -> EngineCapabilities: ...   # can_train, can_embed, ...
```

| Platform | Driver | Inference | On-device training |
|----------|--------|-----------|--------------------|
| **macOS (Apple Silicon)** | `MLXDriver` | MLX / mlx-lm on Metal | **Yes** — QLoRA |
| **Windows / Linux** | `LlamaCppDriver` | llama.cpp (CPU/CUDA/Vulkan) | Limited — train on desktop GPU via HF/PEFT path, or sync adapter |
| **iOS / Android** | `LlamaCppDriver` (mobile build) | llama.cpp / MediaPipe / MLC | **No** — inference only; sync adapter from desktop |

`capabilities.can_train` lets the UI hide/disable the training panel on platforms
that can't train. Phones become **inference clients** that consume adapters the
desktop produced.

**Delivered in this scaffold:** `MLXDriver` is implemented against `mlx-lm`.
`LlamaCppDriver` is stubbed with a working interface and clear `TODO`s so the
Windows/mobile path compiles and runs (returning informative "not implemented"
errors) until you fill it in.

---

## 4b. The voice layer (natural, long-form speech)

Aria can **speak its replies in a natural voice, offline, for as long as the
answer runs**. This is a second pluggable driver layer, structured exactly like
the engine layer, plus a small orchestration piece that makes long speech feel
continuous rather than clip-by-clip.

```python
class VoiceDriver(ABC):
    def synth(self, text, voice=None, rate=1.0) -> SynthResult: ...   # → audio bytes
    def stream(self, segments, ...) -> Iterator[SynthResult]: ...
    @property
    def capabilities(self) -> VoiceCapabilities: ...  # neural, offline, voices, ...
```

| Backend | Driver | Quality | Needs |
|---------|--------|---------|-------|
| **Kokoro-82M** (default on Mac) | `KokoroDriver` | Neural, human-like prosody | `pip install kokoro soundfile`, `brew install espeak-ng`; ~330 MB weights cached in `~/.aria/voices` |
| **macOS `say`** | `SayDriver` | Good (system neural voices) | Nothing — ships with macOS |
| **Fake** | `FakeVoiceDriver` | Silent WAV | Nothing — for tests / headless boxes |

**What makes long speech natural — three moving parts:**

1. **`SentenceChunker`** converts the LLM's *token stream* into a *stream of
   complete sentences*. It splits on real sentence boundaries only — never
   inside abbreviations (`Dr.`, `p.m.`), decimals (`3.14`), or initials
   (`J. R.`) — merges sub-`min_chars` fragments so short interjections aren't
   choppy, and hard-wraps a runaway sentence at the last clause before
   `max_chars`. TTS therefore starts on the **first finished sentence** (~1 s),
   not after the whole answer.
2. **`SpeechSession`** synthesises on a background worker thread and pushes
   finished audio segments onto a queue while later sentences are still being
   generated. The UI plays segments back-to-back → gapless speech of unbounded
   length.
3. **Barge-in.** `stop()` sets a cancel flag the worker checks between segments
   and drains both queues, so Aria goes silent the instant the user interrupts
   or sends a new message. Starting a new turn auto-cancels the previous one.

Playback itself lives in the Tauri webview (HTML5 `Audio`), keeping this whole
layer pure-Python and testable in CI. Endpoints: `GET /voice`,
`POST /voice/set`, `POST /speak`, `POST /voice/stop`, and `POST /chat/speak`
(chat + speak in one call). Backend is chosen by `auto_voice_name()` (prefers
Kokoro → `say` → fake) or pinned with the `ARIA_VOICE` env var.

**Delivered in this scaffold:** `SentenceChunker`, `SpeechSession`/`SpeechManager`,
`FakeVoiceDriver`, and the full API + UI are implemented and tested (30 tests).
`KokoroDriver` and `SayDriver` are written against their documented APIs;
verify-and-run on the Mac (no audio device in the build sandbox).

---

## 5. The self-improvement loop

Two mechanisms, deliberately separated:

### Mechanism A — Memory (instant, always-on)
Every document/conversation is chunked, embedded, and stored in the vector DB. At
query time, relevant chunks are retrieved and injected into the prompt. Updates
the instant you add data; never corrupts the model. **This is where facts live.**

### Mechanism B — Periodic LoRA fine-tuning (the "self-training")
```
   user feedback (👍 / edit / correction)
              │
              ▼
   ┌──────────────────────┐   N new validated
   │  Example store       │──  examples reached ──┐
   │  (instruction→pref)  │                       │
   └──────────────────────┘                       ▼
                                        ┌────────────────────┐
                                        │  Trainer           │
                                        │  split train/held  │
                                        │  QLoRA via engine  │
                                        └─────────┬──────────┘
                                                  ▼
                                        ┌────────────────────┐
                                        │  Eval-and-promote   │
                                        │  gate               │
                                        │  new > active ?     │
                                        │   yes → promote     │
                                        │   no  → discard     │
                                        └─────────┬──────────┘
                                                  ▼
                                        ┌────────────────────┐
                                        │  Adapter registry   │
                                        │  versioned, rollback│
                                        └────────────────────┘
```

**Fine-tuning teaches behavior/style/format — not facts.** Facts go in memory.
The eval gate is the safety mechanism: without it a self-training loop drifts and
degrades. Retraining is **batched**, never per-message (per-message weight updates
cause catastrophic forgetting).

Training target: **Gemma 4 E4B** (fits in 24 GB during QLoRA, batch 1, 4K seq).
The 12B is used for inference; E4B for on-device training.

---

## 6. Data flow: a single chat turn

```
1. UI sends chat(message) → sidecar
2. Memory.retrieve(message)  → top-k relevant chunks
3. Build prompt: system + retrieved context + history + message
4. Engine.generate(...)      → streamed tokens back to UI
5. If the model emits a tool call → Tools.dispatch(), log it, feed result back
6. Response shown; user may give feedback (👍/edit)
7. Feedback.capture(...)     → example store
8. When threshold hit → Trainer runs in background → eval gate → maybe promote
```

---

## 7. Storage layout (on disk)

```
~/.aria/
├── models/                 # downloaded Gemma 4 weights (per size)
│   ├── gemma-4-e4b/
│   └── gemma-4-12b/
├── adapters/               # LoRA adapters, one dir per version
│   ├── v0001/  ...  v000N/
├── aria.db                 # SQLite: memory meta, examples, adapters, tool_calls
├── vectors/                # vector store (LanceDB/Chroma) files
├── voices/                 # Kokoro TTS weights (cached on first run, ~330 MB)
└── config.json
```

See `schemas.md` for the exact table/collection definitions.

---

## 8. Security & privacy posture

- **No telemetry, no cloud.** All inference and training are local.
- **Tool sandboxing.** The shell/exec tool is opt-in and disabled by default;
  every tool call is logged with args + result for audit (see Tools panel).
- **Data ownership.** Everything lives under `~/.aria/`; deleting it removes all
  personal data. Memory entries are individually deletable from the UI.

---

## 9. What's platform-independent vs. platform-specific

| Platform-independent (built & tested here) | Platform-specific (runs on your Mac) |
|---|---|
| Memory / RAG logic | MLX inference + QLoRA training |
| Feedback + example pipeline | Gemma 4 weight download |
| Trainer orchestration + eval gate | Metal GPU execution |
| Tool registry + dispatch + logging | Tauri `.app` compilation |
| Sidecar JSON-RPC API | Native window / OS integration |
| Inspector UI (HTML/JS) | Code-signing / notarization |
