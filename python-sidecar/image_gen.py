"""Local image generation via mflux (FLUX.2 Klein, MLX-native on Apple
Silicon).

Calls mflux's `Flux2Klein` model class and `ImageUtil.save_image` directly,
in-process, rather than shelling out to its CLI entry point
(`mflux-generate-flux2`). That CLI is a console_script installed as a text
file into *this* dev venv's own bin/, with a shebang pointing at this venv's
own python3 — it does not exist in the frozen, shipped app (PyInstaller
bundles this sidecar's own code and its imports into one binary; it does not
carry along another package's unrelated console scripts, nor the interpreter
their shebang lines point to). Reading mflux's own CLI source
(models/flux2/cli/flux2_generate.py) shows its `main()` is just an argparse
wrapper around exactly two calls — `Flux2Klein(...).generate_image()` then
`ImageUtil.save_image()` — both ordinary public methods with no `sys.exit()`
anywhere in that path, so calling them directly in-process is exactly as
solid a contract as depending on the CLI would have been, minus the "does the
script even exist on PATH" problem.

Runs on whatever thread the caller invokes `generate()` from — this module
does no thread management of its own. app.py routes every call through its
single shared MLX GPU worker thread (`SidecarService._gpu_executor`, also
used for the chat engine and STT): MLX binds its Metal command stream to
whichever thread first touches the GPU, so calling this from an arbitrary
HTTP-handler thread would crash the moment it differs from whichever thread
last touched MLX. See app.py's `_EngineThreadProxy` docstring for the full
reasoning; this module intentionally stays agnostic of it so it can be
tested and reasoned about on its own.

Model: mlx-community/flux2-klein-4b-4bit, not the "default" FLUX.1-schnell.
Confirmed live: FLUX.1-schnell's upstream repo on HuggingFace is gated
(requires a logged-in, terms-accepted account) — a dead end for this app,
which has no HF authentication flow anywhere. This one is public, already
MLX-quantized by the same mlx-community org this app already trusts for its
chat models (gemma-4-*), ~4.3GB, and generates a 512x512 image in roughly
15 seconds using ~5GB peak memory on Apple Silicon — confirmed by an actual
generation run, not assumed from documentation.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Optional

MODEL_REPO = "mlx-community/flux2-klein-4b-4bit"

_available: Optional[bool] = None
_model = None  # cached Flux2Klein instance — loaded once, reused across calls


class ImageGenError(RuntimeError):
    pass


def is_available() -> bool:
    """mflux is an optional dependency (see pyproject.toml's `images`
    extra) — the app must still boot and chat fine without it, same as
    kokoro/lancedb/mlx-audio being individually optional. Cached: importing
    mflux drags in torch, not something to redo on every check."""
    global _available
    if _available is None:
        try:
            import mflux  # noqa: F401
            _available = True
        except ImportError:
            _available = False
    return _available


def _get_model():
    global _model
    if _model is None:
        from mflux.models.common.config.model_config import ModelConfig
        from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
        # ModelConfig.flux2_klein_4b() hardcodes the *unquantized* upstream
        # repo (black-forest-labs/FLUX.2-klein-4B). from_name(MODEL_REPO)
        # instead resolves via mflux's substring-inference rule (the repo id
        # contains the "flux2-klein-4b" alias), which builds a config that
        # actually points at MODEL_REPO while inheriting the right
        # architecture — the same resolution the CLI's `--model <repo>` flag
        # goes through.
        _model = Flux2Klein(model_config=ModelConfig.from_name(MODEL_REPO))
    return _model


def _save_image(image, path: str) -> None:
    from mflux.utils.image_util import ImageUtil
    ImageUtil.save_image(image=image, path=path)


def generate(prompt: str, images_dir: str, width: int = 512, height: int = 512,
             steps: int = 4, seed: Optional[int] = None) -> dict:
    """Generates one image, saves it under images_dir, returns
    {"path", "filename", "seconds", "seed"}. Raises ImageGenError (missing
    mflux, empty prompt, generation failure) — the caller turns that into a
    clean error response, never a stack trace.

    Must be called from the app's single MLX GPU worker thread (see module
    docstring) — this function itself does no thread management.
    """
    if not is_available():
        raise ImageGenError(
            "image generation isn't available in this build (mflux not installed)")
    prompt = (prompt or "").strip()
    if not prompt:
        raise ImageGenError("prompt cannot be empty")

    os.makedirs(images_dir, exist_ok=True)
    filename = f"{uuid.uuid4()}.png"
    out_path = os.path.join(images_dir, filename)
    seed = seed if seed is not None else int(time.time())

    t0 = time.time()
    try:
        model = _get_model()
        image = model.generate_image(
            seed=seed, prompt=prompt, num_inference_steps=steps,
            height=height, width=width,
        )
        _save_image(image, out_path)
    except ImageGenError:
        raise
    except Exception as e:
        raise ImageGenError(f"image generation failed: {e}")
    return {
        "path": out_path,
        "filename": filename,
        "seconds": round(time.time() - t0, 1),
        "seed": seed,
    }
