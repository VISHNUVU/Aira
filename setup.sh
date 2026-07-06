#!/usr/bin/env bash
# Aria setup — creates a Python venv and installs the right backend.
set -euo pipefail
cd "$(dirname "$0")"

echo "▸ Aria setup"

# --- pick backend by platform ---------------------------------------------
EXTRA="llamacpp,memory"
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  EXTRA="mlx,memory,voice"
  echo "  detected Apple Silicon → MLX backend + neural voice"
  # espeak-ng powers Kokoro's fallback phonemiser for out-of-dictionary words.
  if command -v brew >/dev/null 2>&1 && ! command -v espeak-ng >/dev/null 2>&1; then
    echo "▸ installing espeak-ng (for the Kokoro voice)"
    brew install espeak-ng || echo "  (skip: install espeak-ng manually if the neural voice errors)"
  fi
else
  echo "  non-Mac platform → llama.cpp backend (stub); voice uses the OS/fallback"
fi

# --- python venv ----------------------------------------------------------
cd python-sidecar
if [[ ! -d .venv ]]; then
  echo "▸ creating virtualenv (.venv)"
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null

echo "▸ installing python-sidecar[$EXTRA]"
pip install -e ".[$EXTRA]" || {
  echo "  full install failed (backend deps may need a Mac). Installing dev/test extras only."
  pip install -e ".[dev]"
}

echo
echo "✓ Done. Next:"
echo "    source python-sidecar/.venv/bin/activate"
echo "    huggingface-cli download mlx-community/gemma-4-12b-it-4bit"
echo "    python python-sidecar/app.py --engine mlx --model gemma-4-12b"
echo
echo "  Or run the tests (no weights needed):  ./run_tests.sh"
