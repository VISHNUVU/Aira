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

# Install the sidecar itself so its modules import, plus PyInstaller.
# Try progressively narrower extras so one broken/unavailable package (e.g.
# kokoro not yet supporting a new Python release) doesn't sacrifice the real
# MLX backend — only degrade voice, or as a last resort the whole backend.
if pip install -e ".[mlx,memory,voice]" >/dev/null 2>&1; then
  ok "installed sidecar with full backend (mlx, memory, voice)"
elif pip install -e ".[mlx,memory]" >/dev/null 2>&1; then
  echo "  voice extras unavailable (e.g. kokoro doesn't support this Python yet) — keeping mlx+memory, TTS falls back to macOS 'say'"
  ok "installed sidecar with mlx + memory backend (no neural voice)"
else
  echo "  full backend install failed — building lean sidecar (fake/say engines)"
  pip install -e ".[dev]" >/dev/null 2>&1 || pip install -e "." >/dev/null 2>&1 || true
  ok "installed sidecar (lean)"
fi
pip install "pyinstaller>=6" >/dev/null
ok "PyInstaller ready"

# Keep PyInstaller's cache inside the repo (clean, deletable, sandbox-safe).
export PYINSTALLER_CONFIG_DIR="$SIDECAR_DIR/.pyi-cache"
mkdir -p "$PYINSTALLER_CONFIG_DIR"

echo "  running PyInstaller (aria-sidecar.spec)…"
pyinstaller aria-sidecar.spec --noconfirm --clean >/dev/null
[[ -f "$SIDECAR_DIR/dist/aria-sidecar" ]] || die "PyInstaller did not produce dist/aria-sidecar"
ok "sidecar frozen: $(du -h dist/aria-sidecar | cut -f1) Mach-O binary"
deactivate
cd "$ROOT"

# --- 2. stage the binary at the externalBin triple path --------------------
say "Staging sidecar for Tauri (externalBin)"
mkdir -p "$BIN_DIR"
cp "$SIDECAR_DIR/dist/aria-sidecar" "$STAGED_BIN"
chmod +x "$STAGED_BIN"
ok "staged $STAGED_BIN"

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
say "Building Aria.app + .dmg (tauri build — first Rust compile is slow)"
# --target makes the bundle identifier explicit and matches the sidecar triple.
npx tauri build --target "$TRIPLE"

# --- 6. report artifacts ----------------------------------------------------
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
