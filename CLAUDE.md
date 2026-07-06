# Handoff notes for Claude Code (finish Aria on your Mac)

This project was scaffolded and tested in a Linux sandbox where **all
platform-independent logic is already built and passing 42/42 tests** using a
`fake` engine driver (no model weights). What remains is everything that needs a
real Apple Silicon Mac: downloading weights, running MLX/Metal inference and
QLoRA, and compiling the `.app`.

Open this folder in **Claude Code on your Mac** and work through the tasks below.
Each one is scoped and verifiable.

---

## Ground truth: what already works

Run this first to confirm the logic layer is intact:

```bash
./run_tests.sh        # expect: 42/42 passing, no weights needed
```

These modules are done and tested — **don't rewrite them, just wire the real
engine underneath**:
- `memory.py` (RAG), `feedback.py`, `trainer.py`, `eval_gate.py`, `tools.py`,
  `store.py`, `app.py` (HTTP API), and the whole `src/` web UI.
- The `EngineDriver` ABC in `engine/base.py` is the contract. `fake_driver.py`
  implements it for tests; `mlx_driver.py` is the real one to finish.

---

## Task 1 — Environment + weights (30 min)

```bash
./setup.sh
source python-sidecar/.venv/bin/activate
python -c "import mlx.core as mx; print('metal:', mx.metal.is_available())"
huggingface-cli download mlx-community/gemma-4-e4b-it-4bit   # training target
huggingface-cli download mlx-community/gemma-4-12b-it-4bit   # chat model
```

**Verify:** `mx.metal.is_available()` prints `True`.

## Task 2 — Verify & run the MLX driver (it's already written)

`engine/mlx_driver.py` is a **complete implementation** against the documented
mlx-lm API (verified 2026) — not a skeleton. All five methods are filled in.
It has just never executed on Metal (no Apple GPU in the build sandbox). Your
job here is **verify-and-run**, not write-from-scratch. Read it once, then
confirm each method against your installed `mlx_lm` version:

- `load()` → uses `mlx_lm.load(model_path, adapter_path=...)`, keeps the
  tokenizer for chat templating. ✓ written
- `generate()` → applies the Gemma 4 chat template to `messages` (passing
  `tools=` for native function calling) and streams via `stream_generate`,
  yielding `resp.text`. **Note the deliberate `sampler=make_sampler(temp=...)`
  — passing `temp=` straight to `generate()` raises `TypeError`.** ✓ written
- `embed()` → loads an MLX embedding model via `mlx_embeddings`. **Confirm the
  model's output dim matches `embed_dim` in `app.py` (768)** — if you pick a
  different embedder, update both. ✓ written
- `train_lora()` → shells out to `mlx_lm.lora` with a train/valid split dir
  (`_prepare_data_dir` builds it), parses losses, returns `TrainResult`. ✓ written
- `set_adapter()` → reloads the model with `adapter_path=`. ✓ written

The two things most likely to need a tweak on your machine: (a) the exact
`mlx_embeddings` import/return shape for your chosen embed model, and (b) the
`mlx_lm.lora` CLI flag names if you're on a newer mlx-lm (run
`python -m mlx_lm.lora --help` and diff against the `cmd` list in `train_lora`).

**Verify after each:**
```bash
python -c "from engine import make_engine; e=make_engine('mlx','~/.aria/models','~/.aria/adapters'); e.load('gemma-4-12b'); print(''.join(e.generate([{'role':'user','content':'hi'}], stream=False)))"
```
Then re-run `./run_tests.sh` but with `--engine mlx` where relevant — the
**integration test logic is identical**, only the driver changes. The cleanest
check: temporarily point `SidecarService(engine_name="mlx")` and hit `/status`
and `/chat`.

## Task 3 — Real end-to-end training run

With E4B downloaded and >5 feedback examples captured (use the Training tab or
POST `/feedback`), click **Train now**. Confirm:
- an adapter appears under `~/.aria/adapters/vNNNN/`,
- the eval gate logs a candidate-vs-baseline score,
- promotion only happens if it wins (check the Adapters tab).

The gate logic is already proven in `tests/test_train_loop.py`; here you're just
confirming the *real* MLX train + eval produces sane numbers. If scores look
random, check that `embed()` is deterministic and dimensioned correctly.

## Task 4 — Wire LanceDB (optional, 15 min)

`memory.py` already has `LanceVectorStore` with an InMemory fallback. Just
`pip install lancedb` and confirm `SidecarService(use_lance=True)` picks it up
(it falls back silently if the import fails). Verify vectors persist across
restarts.

## Task 4b — Verify the natural voice (Kokoro) — ~15 min

The whole voice pipeline (sentence chunking → speech session → `/speak`,
`/chat/speak`, `/voice/stop` endpoints → UI playback + Stop button) is **built
and tested** on the `fake` voice (30 tests). Two driver files render the actual
audio and need a real machine to run:

- `voice/kokoro_driver.py` — neural, offline, the default. **This is the one
  that makes Aria "speak naturally."**
- `voice/say_driver.py` — macOS `say`, zero-dependency fallback.

Steps:
```bash
pip install kokoro soundfile          # or:  pip install -e ".[voice]"
brew install espeak-ng                 # phonemiser for OOV words
# smoke test the neural voice end-to-end:
python -c "from voice import make_voice; r=make_voice('kokoro').synth('Hello, I am Aria, running entirely on your Mac.'); open('/tmp/aria.wav','wb').write(r.audio); print(r.seconds,'s')"
afplay /tmp/aria.wav
```
Two things most likely to need a tweak: (a) the `KPipeline` import path / voice
id if your `kokoro` version differs (`af_heart` is the default; `list_voices()`
shows the set), and (b) confirm `auto_voice_name()` picks `kokoro` once it's
installed. If Kokoro won't install, the app still speaks via `say` — set
`ARIA_VOICE=say` (or leave `auto`, which falls back). In the UI: **Models tab →
Voice** to pick a voice, set speed, and hit **Test voice**; the **🔊 Speak
replies** toggle in Chat turns on spoken answers, and **◼ Stop** is barge-in.

Honest framing to keep: the voice is **synthesis only** — it reads Aria's text
answers aloud. It is independent of memory/RAG (facts) and QLoRA (behaviour);
nothing about speaking changes the model or gets trained into weights.

## Task 5 — Build the desktop app  ✅ DONE (pipeline built)

The full distribution pipeline is wired. One command:

```bash
scripts/build-macos.sh     # freeze sidecar → stage → icons → tauri build → .dmg
```

produces `Aria.app` + `Aria_0.1.0_aarch64.dmg` under
`src-tauri/target/aarch64-apple-darwin/release/bundle/`. What's already in place:

- **Frozen sidecar** — `python-sidecar/aria-sidecar.spec` (PyInstaller onefile).
  One spec serves both the lean build (fake/say engines) and the full build
  (folds in mlx/lancedb/kokoro/etc. only if importable). `console=True` is
  required — the `ARIA_PORT=` handshake goes over stdout.
- **externalBin wiring** — `tauri.conf.json` → `externalBin: ["binaries/aria-sidecar"]`;
  the binary is staged at `src-tauri/binaries/aria-sidecar-aarch64-apple-darwin`
  (Tauri appends the target triple). `main.rs` spawns it via
  `app.shell().sidecar("aria-sidecar")`, with a dev fallback to
  `python3 ../python-sidecar/app.py`. `capabilities/default.json` grants
  `shell:allow-spawn` for the sidecar + python3 fallback.
- **Icons** — `src-tauri/icons/` (icns + PNG set), regenerable via
  `scripts/make-icons.py` from `icon_source_1024.png`.
- **Signing (optional)** — `scripts/sign-and-notarize.sh` + `docs/DISTRIBUTION.md`.

Dev loop is unchanged: `npm run tauri dev` spawns the sidecar from source.

**Verify:** run `scripts/build-macos.sh` on your Mac (needs Rust + Node), then
`open` the built `Aria.app` — status card shows `online`, chat works. See
`docs/DISTRIBUTION.md` for the signed/notarized path and troubleshooting.

---

## Guardrails / things not to break

- **Keep the honest framing.** Memory = facts (instant, RAG). QLoRA = behavior
  (periodic, gated). Never train per-message. Never promote an adapter that
  didn't beat baseline — the gate in `eval_gate.py` is the safety-critical piece.
- **`run_shell` tool ships disabled.** It's a real command runner; keep it
  off-by-default and behind an explicit user toggle.
- The `EngineDriver` contract is the seam. If you need new capabilities, add
  them to the ABC + the `fake` driver (so tests still run without weights),
  then implement in `mlx_driver.py`.
- Storage lives under `~/.aria/` (`config.json`, `aria.db`, `models/`,
  `adapters/`, `vectors/`).

## The API surface (already built — the UI depends on it)

`GET /status` · `POST /chat` · `GET|POST /memory` · `POST /memory/delete` ·
`POST /feedback` · `POST /train` · `GET /training/runs` · `GET /examples` ·
`GET /adapters` · `POST /adapters/activate` · `POST /adapters/rollback` ·
`GET /tools` · `POST /tools/toggle` · `GET /tools/calls` · `POST /tools/invoke` ·
`GET /models` · `POST /models/download` ·
`GET /voice` · `POST /voice/set` · `POST /speak` · `POST /voice/stop` ·
`POST /chat/speak`

All return JSON; see `_route()` in `app.py` for exact request/response shapes.
Voice storage lives under `~/.aria/voices/` (Kokoro weights, cached on first run).
