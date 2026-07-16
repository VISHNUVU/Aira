#!/usr/bin/env bash
# ============================================================================
# build-macos.sh — one command, unsigned installable Aria.app + .dmg
# ============================================================================
# Produces a distributable macOS app for Apple Silicon:
#   1. Freezes the Python sidecar into a standalone binary (PyInstaller)
#   2. Stages it at the Tauri externalBin target-triple path
#   3. Ensures the app icon set exists
#   4. Installs JS deps and compiles the Tauri (Rust) shell
#   5. Emits Aria.app and a .dmg under src-tauri/target/release/bundle/
#
# The result is UNSIGNED: end users open it with right-click -> Open the first
# time (Gatekeeper). To ship without that warning, run scripts/sign-and-notarize.sh
# afterwards with your Apple Developer ID (see docs/DISTRIBUTION.md).
#
# Requirements on the BUILD machine (an Apple Silicon Mac):
#   - Xcode Command Line Tools  (xcode-select --install)
#   - Rust + Cargo              (https://rustup.rs)
#   - Node.js + npm             (https://nodejs.org, or `brew install node`)
#   - Python 3.11+              (system python3 is fine)
# Everything else (Tauri CLI, PyInstaller) is installed automatically below.
# ============================================================================
set -euo pipefail

# --- locate ourselves -------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

TRIPLE="aarch64-apple-darwin"          # Apple Silicon target triple
SIDECAR_DIR="$ROOT/python-sidecar"
BIN_DIR="$ROOT/src-tauri/binaries"
STAGED_BIN="$BIN_DIR/aria-sidecar-$TRIPLE"

say()  { printf '\n\033[1;34m▸ %s\033[0m\n' "$*"; }
ok()   { printf '  \033[0;32m✓ %s\033[0m\n' "$*"; }
die()  { printf '\n\033[0;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# --- 0. sanity: platform + toolchain ---------------------------------------
say "Checking build host"
[[ "$(uname -s)" == "Darwin" ]]  || die "macOS required (this builds a .app/.dmg)."
[[ "$(uname -m)" == "arm64" ]]   || die "Apple Silicon required (MLX + arm64 sidecar)."
command -v python3 >/dev/null    || die "python3 not found."
command -v cargo   >/dev/null    || die "Rust/Cargo not found. Install: https://rustup.rs"
command -v npm     >/dev/null    || die "Node/npm not found. Install: https://nodejs.org"
ok "Apple Silicon macOS, python3 + cargo + npm present"

# APP_VERSION (python-sidecar/app.py) and tauri.conf.json's "version" are two
# separately hand-maintained strings with a comment on each saying "keep in
# lockstep" — nothing previously enforced that, so a future release could
# silently ship with the sidecar and the .app bundle's Info.plist reporting
# two different version numbers. Fail loudly here instead.
SIDECAR_VERSION="$(python3 -c "import re; print(re.search(r'APP_VERSION = \"([^\"]+)\"', open('$SIDECAR_DIR/app.py').read()).group(1))")"
TAURI_VERSION="$(python3 -c "import json; print(json.load(open('$ROOT/src-tauri/tauri.conf.json'))['version'])")"
[[ "$SIDECAR_VERSION" == "$TAURI_VERSION" ]] || die \
  "version drift: python-sidecar/app.py's APP_VERSION ($SIDECAR_VERSION) != src-tauri/tauri.conf.json's version ($TAURI_VERSION) — fix one before building"
ok "version in sync: $SIDECAR_VERSION"

# --- 1. freeze the Python sidecar ------------------------------------------
say "Freezing Python sidecar (PyInstaller)"
cd "$SIDECAR_DIR"
if [[ ! -d .venv ]]; then
  echo "  creating build venv (.venv)"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null

# Reproducible builds: requirements.lock is a pip freeze of a venv that
# produced a fully verified release (model loads, chat works, gate passed).
# Installing from it — exact pins, no resolver — means two builds from the
# same commit get the same dependency set, instead of drifting with
# whatever pip resolves that day (confirmed harm: a stray opencv-python in
# the venv once added ~140MB of never-imported video codecs to the bundle,
# and an unpinned transformers upgrade would break mlx-lm outright).
# The legacy extras-based path below remains only as a fallback for a
# checkout where the lockfile is missing; regenerate the lockfile from a
# verified venv with:  pip freeze --exclude-editable > requirements.lock
HAVE_MLX=0
if [[ -f requirements.lock ]]; then
  echo "  installing pinned dependency set (requirements.lock)"
  pip install --no-deps -q -r requirements.lock || die "locked dependency install failed — fix requirements.lock rather than shipping a drifted build"
  pip install --no-deps -q -e . >/dev/null
  pip uninstall -y hf-xet >/dev/null 2>&1 || true
  python -c "import mlx.core" >/dev/null 2>&1 && HAVE_MLX=1
  ok "installed locked dependency set ($(wc -l < requirements.lock | tr -d ' ') pins, mlx=$HAVE_MLX)"
else
  echo "  WARNING: requirements.lock missing — falling back to unpinned resolution"
  if pip install -e ".[mlx,memory,voice]" >/dev/null 2>&1; then
    ok "installed sidecar with full backend (mlx, memory, voice)"
    HAVE_MLX=1
  elif pip install -e ".[mlx,memory]" >/dev/null 2>&1; then
    echo "  voice extras unavailable — keeping mlx+memory, TTS falls back to macOS 'say'"
    HAVE_MLX=1
  else
    echo "  full backend install failed — building lean sidecar (fake/say engines)"
    pip install -e ".[dev]" >/dev/null 2>&1 || pip install -e "." >/dev/null 2>&1 || true
  fi
  if [[ "$HAVE_MLX" == "1" ]]; then
    pip install --no-deps "mlx-vlm>=0.6" "mlx-audio>=0.4" >/dev/null 2>&1 || true
    pip install -q "transformers==5.0.0" >/dev/null 2>&1
    pip uninstall -y hf-xet >/dev/null 2>&1 || true
    # Local image generation (mflux): unlike mlx-vlm/mlx-audio above, this
    # genuinely needs its own full dependency resolution (torch et al are
    # functional requirements, not something to --no-deps around) — a
    # deliberate, accepted exception to the MLX-only policy (see
    # pyproject.toml's `images` extra and image_gen.py's module docstring).
    pip install -q ".[images]" >/dev/null 2>&1 || echo "  mflux install failed — building without image generation"
  fi
  pip install "pyinstaller>=6" >/dev/null
fi
ok "PyInstaller ready"

# Keep PyInstaller's cache inside the repo (clean, deletable, sandbox-safe).
export PYINSTALLER_CONFIG_DIR="$SIDECAR_DIR/.pyi-cache"
mkdir -p "$PYINSTALLER_CONFIG_DIR"

echo "  running PyInstaller (aria-sidecar.spec)…"
pyinstaller aria-sidecar.spec --noconfirm --clean >/dev/null
[[ -f "$SIDECAR_DIR/dist/aria-sidecar/aria-sidecar" ]] || die "PyInstaller did not produce dist/aria-sidecar/aria-sidecar (onedir)"
ok "sidecar frozen (onedir): $(du -sh dist/aria-sidecar | cut -f1) total"
deactivate
cd "$ROOT"

# --- 2. stage the onedir build for Tauri ------------------------------------
# The executable goes through Tauri's normal externalBin convention; its
# `_internal/` dependency tree (everything PyInstaller unpacked ahead of
# time instead of at launch — see aria-sidecar.spec's onedir note) ships as
# a bundled Resource instead, since externalBin only copies a single file.
# main.rs symlinks the two together next to each other on first launch.
say "Staging sidecar for Tauri (onedir: externalBin + resources)"
mkdir -p "$BIN_DIR"
cp "$SIDECAR_DIR/dist/aria-sidecar/aria-sidecar" "$STAGED_BIN"
chmod +x "$STAGED_BIN"
ok "staged executable: $STAGED_BIN"

SIDECAR_INTERNAL_DIR="$ROOT/src-tauri/resources/sidecar-internal"
rm -rf "$SIDECAR_INTERNAL_DIR"
mkdir -p "$SIDECAR_INTERNAL_DIR"
cp -R "$SIDECAR_DIR/dist/aria-sidecar/_internal" "$SIDECAR_INTERNAL_DIR/_internal"
ok "staged dependency tree: $SIDECAR_INTERNAL_DIR/_internal ($(du -sh "$SIDECAR_INTERNAL_DIR" | cut -f1))"

# --- 3. ensure icons exist --------------------------------------------------
say "Checking app icons"
if [[ -f "$ROOT/src-tauri/icons/icon.icns" ]]; then
  ok "icon set present"
else
  if [[ -f "$ROOT/icon_source_1024.png" ]]; then
    echo "  regenerating icons from icon_source_1024.png"
    python3 "$SCRIPT_DIR/make-icons.py" || die "icon generation failed"
    ok "icons generated"
  else
    die "no icons and no icon_source_1024.png to build them from"
  fi
fi

# --- 4. JS deps + Tauri CLI -------------------------------------------------
say "Installing JS dependencies"
npm install >/dev/null 2>&1 || npm install
ok "node_modules ready"

# The Tauri CLI comes from devDependencies; make sure the binary resolves.
if ! npx --no-install tauri --version >/dev/null 2>&1; then
  echo "  installing @tauri-apps/cli"
  npm install --save-dev @tauri-apps/cli@^2 >/dev/null 2>&1 || true
fi

# --- 5. compile the app + bundle -------------------------------------------
say "Building Aria.app (tauri build — first Rust compile is slow)"
# Belt-and-suspenders: clear every place a stale _internal has been found
# sitting between builds, so a fresh build never inherits anything from a
# previous one regardless of which of these actually matters.
rm -rf "$ROOT/src-tauri/target/$TRIPLE/release/bundle/macos/Aria.app"
rm -rf "$ROOT/src-tauri/target/$TRIPLE/release/_internal"
rm -rf "$ROOT/src-tauri/target/debug/_internal"
npx tauri build --target "$TRIPLE" --bundles app

# --- 5b. inject the sidecar dependency tree + package the dmg ourselves -----
# ROOT CAUSE (definitively isolated, end of a long session): Tauri's
# `resources` bundler does not preserve symlinks — it flattens
# Python.framework's internal structure (Python -> Versions/Current/Python
# became a dereferenced regular file; Versions/Current and Resources links
# were dropped entirely; verified via diff -rq between PyInstaller's pristine
# onedir output and what Tauri shipped). A Python.framework without its
# version symlinks breaks MLX's Metal shader library resolution with
# `RuntimeError: Failed to load the default metallib`, killing model loading
# in the packaged app while the identical unpackaged build worked fine.
# Proof both ways: hand-copying the same _internal into the same bundle with
# plain `cp -R` (which preserves symlinks) and running the same binary
# in-place loaded gemma-4-12b in seconds, with no other change.
# So `_internal` is deliberately NOT in tauri.conf.json's `resources` — it's
# copied in by hand here, and the dmg is packaged with hdiutil directly
# (tauri build --bundles dmg can't be used for it: it re-cleans and
# re-bundles Aria.app from scratch first, wiping this injection).
say "Injecting sidecar dependency tree (symlink-preserving)"
APP_PATH="$ROOT/src-tauri/target/$TRIPLE/release/bundle/macos/Aria.app"
[[ -d "$APP_PATH" ]] || die "tauri build did not produce $APP_PATH"
rm -rf "$APP_PATH/Contents/Resources/_internal"
cp -R "$SIDECAR_INTERNAL_DIR/_internal" "$APP_PATH/Contents/Resources/_internal"
# Fail the build outright if the framework symlinks didn't survive the copy —
# this exact breakage cost a full day to find; never let it ship silently.
[[ -L "$APP_PATH/Contents/Resources/_internal/Python.framework/Python" ]] \
  || die "Python.framework symlinks lost during copy — build would ship broken"
ok "injected _internal ($(du -sh "$APP_PATH/Contents/Resources/_internal" | cut -f1)), framework symlinks intact"

say "Packaging .dmg (hdiutil)"
VERSION="$(python3 -c "import json; print(json.load(open('$ROOT/src-tauri/tauri.conf.json'))['version'])")"
DMG_OUT_DIR="$ROOT/src-tauri/target/$TRIPLE/release/bundle/dmg"
mkdir -p "$DMG_OUT_DIR"
rm -f "$DMG_OUT_DIR"/*.dmg
DMG_NAME="Aria_${VERSION}_aarch64.dmg"
hdiutil create -volname "Aria" -srcfolder "$APP_PATH" -ov -format UDZO \
  "$DMG_OUT_DIR/$DMG_NAME" >/dev/null
ok "packaged $DMG_OUT_DIR/$DMG_NAME"

# --- 6. release gate ---------------------------------------------------------
# Every build must prove itself before it can ship: structure (framework
# symlinks, metallib, no bloat), runtime (real launch, real model load, real
# chat), lifecycle (auto-restart, clean quit). See scripts/verify-build.sh.
# Set ARIA_SKIP_GATE=1 only for iterating on the build script itself.
if [[ "${ARIA_SKIP_GATE:-0}" != "1" ]]; then
  say "Release gate (scripts/verify-build.sh)"
  "$SCRIPT_DIR/verify-build.sh" || die "release gate FAILED — this build must not ship"
fi

# --- 7. report artifacts ----------------------------------------------------
BUNDLE_DIR="$ROOT/src-tauri/target/$TRIPLE/release/bundle"
say "Build complete"
APP_PATH="$(/usr/bin/find "$BUNDLE_DIR" -name 'Aria.app' -maxdepth 3 2>/dev/null | head -1 || true)"
DMG_PATH="$(/usr/bin/find "$BUNDLE_DIR" -name '*.dmg'    -maxdepth 3 2>/dev/null | head -1 || true)"
[[ -n "$APP_PATH" ]] && ok "App:  $APP_PATH"
[[ -n "$DMG_PATH" ]] && ok "DMG:  $DMG_PATH"
cat <<EOF

  Next steps:
    • Test locally:   open "$APP_PATH"
    • Distribute:     share the .dmg. First launch on another Mac:
                      right-click Aria.app → Open (unsigned; one-time Gatekeeper prompt).
    • Ship signed:    scripts/sign-and-notarize.sh   (needs an Apple Developer ID)

  On first launch Aria downloads its model (~7 GB) — that is expected and one-time.
EOF
