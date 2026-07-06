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
* onefile keeps distribution simple; the binary self-extracts to a temp dir at
  launch. Startup is a second or two — fine for a desktop app.
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

# numpy is a hard dependency — make sure all of it comes along.
hiddenimports.extend(collect_submodules("numpy"))

# Update-check token (read-only, single-repo-scoped fine-grained GitHub PAT).
# Bundled as data so the frozen app can check for new releases without the
# end user needing to configure anything. Never committed to git — see
# .gitignore. Missing gracefully: update_checker.py treats no-file as
# "update checks disabled", it doesn't fail the build or the app.
import os as _os
_token_path = _os.path.join(_os.path.dirname(_os.path.abspath(SPEC)), ".update_token")
if _os.path.isfile(_token_path):
    datas.append((_token_path, "."))
    print("[aria-sidecar.spec] bundling .update_token")
else:
    print("[aria-sidecar.spec] no .update_token found — update checks will be disabled in this build")


a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Trim obvious dev-only weight so the lean binary stays small.
    excludes=["pytest", "tkinter", "matplotlib", "IPython", "pandas"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="aria-sidecar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,          # sidecar writes the ARIA_PORT= handshake to stdout
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,      # follows the build machine (arm64 on Apple Silicon)
    codesign_identity=None,
    entitlements_file=None,
)
