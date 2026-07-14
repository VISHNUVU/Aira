# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Aria sidecar.

Produces a single standalone binary — `aria-sidecar` — that bundles the Python
runtime and every sidecar module, so end users need no Python installed.

Build:
    pyinstaller aria-sidecar.spec --noconfirm --clean

Output:
    dist/aria-sidecar          (the binary the Tauri shell spawns)

Design notes
------------
* The engine + voice drivers are imported *dynamically* (make_engine() does
  `from .mlx_driver import MLXDriver` at call time), so PyInstaller's static
  analysis can't see them. We list them in `hiddenimports` so they're bundled.
* Heavy backends (mlx, mlx_lm, lancedb, kokoro, soundfile, llama_cpp) are
  OPTIONAL and guarded by try/except ImportError in the source. We do NOT force
  them into the bundle here — a lean sidecar still boots and runs the fake / OS
  `say` / in-memory paths. `collect_optional()` folds each in only if it's
  actually installed in the build venv, so one spec serves both a lean build
  and a full Apple-Silicon build (where you've `pip install .[mlx,memory,voice]`
  before freezing).
* onedir, not onefile: with mlx/mlx_lm/mlx_vlm bundled, the frozen tree is
  several hundred MB. onefile would re-extract all of that into a fresh temp
  dir on *every single launch* (confirmed live: 2GB+ of writes and 15-30s+
  before the sidecar can even start listening) — slow, disk-write-heavy, and
  a plausible source of the intermittent "sidecar never comes up" failures
  seen in testing. onedir launches the real executable directly against its
  already-unpacked `_internal/` dependency tree: no extraction step, no
  per-launch write burst, first byte of output within ~1s. The build script
  ships `_internal/` as a bundled Resource and main.rs symlinks it next to
  the externalBin executable at first launch — see main.rs's
  `ensure_sidecar_internal_symlink()`.
"""
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# --- always bundle: our own dynamically-imported drivers ------------------
hiddenimports = [
    "engine.mlx_driver",
    "engine.llamacpp_driver",
    "engine.fake_driver",
    "voice.kokoro_driver",
    "voice.say_driver",
    "voice.fake_driver",
]

datas = []
binaries = []


def collect_optional(pkg, submodules=True, data=False):
    """Fold an optional dependency into the bundle *iff* it's importable in the
    build venv. Keeps one spec working for both lean and full builds."""
    import importlib.util
    if importlib.util.find_spec(pkg) is None:
        print(f"[aria-sidecar.spec] optional dep not present, skipping: {pkg}")
        return
    print(f"[aria-sidecar.spec] bundling optional dep: {pkg}")
    if submodules:
        hiddenimports.extend(collect_submodules(pkg))
    if data:
        datas.extend(collect_data_files(pkg))


# Apple-Silicon inference + on-device QLoRA (only present in a full build):
# mlx_vlm is required for unified/multimodal Gemma checkpoints (e.g. the 12B
# and e4b mlx-community exports both declare vision_config/audio_config, so
# mlx_driver.py routes them through mlx_vlm, not plain mlx_lm — see
# engine/mlx_driver.py's _is_multimodal_checkpoint()).
for _p in ("mlx", "mlx_lm", "mlx_vlm", "mlx_embeddings"):
    collect_optional(_p, data=True)
# Offline speech-to-text (mic button dictation, Whisper via mlx_audio):
collect_optional("mlx_audio", data=True)
# RAG vector store:
for _p in ("lancedb", "pyarrow"):
    collect_optional(_p, data=True)
# Neural voice:
for _p in ("kokoro", "soundfile"):
    collect_optional(_p, data=True)
# Cross-platform inference:
collect_optional("llama_cpp", data=True)
# Local image generation (image_gen.py) — pulls in torch/transformers, both
# of which ship their own PyInstaller hooks (pyinstaller-hooks-contrib's
# hook-torch.py / hook-transformers.py), so their binaries/data are handled
# automatically once torch/transformers themselves are reachable from mflux's
# own imports; we only need to seed the walk from mflux itself. cv2 and
# matplotlib stay excluded below — confirmed live that the txt2img generation
# path this app actually uses (Flux2Klein.generate_image) never imports
# either; they're only pulled in by mflux's unused concept_attention/
# controlnet variants.
collect_optional("mflux", data=True)
# certifi is a plain top-level hard dependency (update_checker.py,
# web_search.py) but its cacert.pem is *data*, not code — PyInstaller's
# static analysis bundles the module fine on its own but skips this file
# without an explicit collect_data_files, which would silently break the
# HTTPS cert path in a frozen build despite `import certifi` succeeding.
datas.extend(collect_data_files("certifi"))

# numpy is a hard dependency — make sure all of it comes along.
hiddenimports.extend(collect_submodules("numpy"))

# Note: update_checker.py hits a plain public URL (self-hosted manifest) —
# no credential needs bundling for update checks anymore.


a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Trim obvious dev-only weight so the lean binary stays small. cv2/av
    # specifically: confirmed live that opencv-python sitting in the build
    # venv (a stray transitive dependency of something under mlx-audio/
    # mlx-vlm's optional media extras, never imported by our own code) gets
    # swept into the bundle by PyInstaller's automatic analysis purely
    # because it's importable — ~200MB of unrelated video-codec libraries
    # (libavcodec, libx264, etc.) along for the ride, and non-deterministic
    # across builds depending on exactly what happens to be installed in
    # the venv at freeze time. Excluding explicitly keeps the build
    # reproducible regardless of venv drift.
    # twine/keyring: mflux declares them as plain (non-optional) dependencies
    # in its own metadata — presumably for its own maintainers' `twine upload`
    # release process — even though nothing in the actual generation path
    # this app calls (Flux2Klein.generate_image / ImageUtil.save_image) ever
    # imports them. Confirmed live (see image_gen.py's module docstring).
    #
    # pandas was excluded here too until confirmed live (2026-07-13) that
    # it's a genuine, load-bearing runtime dependency for training: HF's
    # `datasets.load_dataset()` (used by mlx_vlm.lora's real-data loading
    # path — see MLXDriver._train_lora_vlm) imports it internally and fails
    # with "No module named 'pandas'" without it. Unlike cv2/twine/keyring,
    # this one is actually needed by a path this app calls.
    excludes=["pytest", "tkinter", "matplotlib", "IPython", "cv2",
              "twine", "keyring"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # onedir: binaries/data go in COLLECT below, not in the exe
    name="aria-sidecar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    console=True,          # sidecar writes the ARIA_PORT= handshake to stdout
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,      # follows the build machine (arm64 on Apple Silicon)
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="aria-sidecar",
)
