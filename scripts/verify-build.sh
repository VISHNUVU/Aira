#!/usr/bin/env bash
# ============================================================================
# verify-build.sh — automated release gate for Aria.app
# ============================================================================
# Nothing ships unless it passes here. Born out of a full day lost to a
# packaged build that looked fine, launched fine, and then silently couldn't
# load its model (Tauri's resource bundler had flattened Python.framework's
# symlinks). Every failure mode discovered that day is a permanent check:
#
#   STRUCTURE  Python.framework symlinks intact, metallib present, no stray
#              cv2/ffmpeg bloat swept in by venv drift
#   RUNTIME    the real .app launches, binds its port, LOADS THE REAL MODEL,
#              and answers a real chat prompt on the GPU
#   LIFECYCLE  sidecar auto-restarts if killed; quit leaves zero orphans
#
# On success writes .release-gate-pass next to the .dmg containing the dmg's
# sha256 — publish-update.sh refuses to upload without a matching stamp.
#
# Usage:
#   scripts/verify-build.sh                  # gate the default build output
#   scripts/verify-build.sh /path/Aria.app   # gate a specific bundle
#
# Notes: needs the gemma-4-12b weights in ~/.aria/models (skips the model
# and chat checks with a loud warning if absent — structure and lifecycle
# still run). Kills any running Aria first: the shell's startup pkill sweep
# would race a survivor anyway.
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRIPLE="aarch64-apple-darwin"
APP="${1:-$ROOT/src-tauri/target/$TRIPLE/release/bundle/macos/Aria.app}"
DMG_DIR="$ROOT/src-tauri/target/$TRIPLE/release/bundle/dmg"
STAMP="$DMG_DIR/.release-gate-pass"

PASS=0; FAIL=0
say()  { printf '\n\033[1;34m▸ %s\033[0m\n' "$*"; }
ok()   { printf '  \033[0;32m✓ %s\033[0m\n' "$*"; PASS=$((PASS+1)); }
warn() { printf '  \033[1;33m! %s\033[0m\n' "$*"; }
bad()  { printf '  \033[0;31m✗ %s\033[0m\n' "$*"; FAIL=$((FAIL+1)); }

cleanup() {
  pkill -9 -f "Aria.app/Contents/MacOS" 2>/dev/null || true
  [[ -n "${GATE_HOME:-}" ]] && rm -rf "$GATE_HOME"
}
trap cleanup EXIT

# --- structure ---------------------------------------------------------------
say "STRUCTURE: bundle integrity"
[[ -d "$APP" ]] || { bad "no bundle at $APP"; exit 1; }

INTERNAL="$APP/Contents/Resources/_internal"
if [[ -L "$INTERNAL/Python.framework/Python" && -L "$INTERNAL/Python.framework/Resources" ]]; then
  ok "Python.framework symlinks intact"
else
  bad "Python.framework symlinks flattened/missing — model loading WILL fail (the root cause of 2026-07-07)"
fi

if [[ -f "$INTERNAL/mlx/lib/mlx.metallib" ]]; then
  ok "mlx.metallib present ($(du -h "$INTERNAL/mlx/lib/mlx.metallib" | cut -f1))"
else
  bad "mlx.metallib missing — Metal inference impossible"
fi

if [[ -e "$INTERNAL/cv2" ]] || find "$INTERNAL" -maxdepth 1 -name "libavcodec*" | grep -q .; then
  bad "stray cv2/ffmpeg libraries in bundle — venv drift leaked past the spec excludes"
else
  ok "no stray cv2/ffmpeg bloat"
fi

[[ -x "$APP/Contents/MacOS/aria-sidecar" ]] && ok "sidecar executable present" || bad "sidecar executable missing"

# The UI discovers the sidecar's random port ONLY through window.__TAURI__
# (invoke("sidecar_port") + the sidecar-ready event), which Tauri v2 injects
# solely when app.withGlobalTauri is true. With it false/absent the webview
# silently falls back to dev port 8765 and every request fails with "Load
# failed" — while the backend, tested from outside via curl, looks perfectly
# healthy. Shipped broken for days exactly this way; never again.
# Detection marker verified empirically: withGlobalTauri isn't stored as
# config text — enabling it embeds the global API JS bundle into the binary
# (+~50KB), whose injected scripts all start with `if("__TAURI__"in window)`.
# A build without that marker has no window.__TAURI__ at runtime, period.
# grep -c (not -q): -q exits on first match, strings takes SIGPIPE, and this
# script's pipefail turns that successful match into a failed pipeline.
TAURI_MARKER="$(strings "$APP/Contents/MacOS/aria" 2>/dev/null | grep -c '__TAURI__.in window' || true)"
if [[ "$TAURI_MARKER" -gt 0 ]]; then
  ok "withGlobalTauri bundle embedded — UI can discover the sidecar port"
else
  bad "window.__TAURI__ injection bundle NOT in binary — packaged UI cannot reach its backend"
fi

# --- runtime -----------------------------------------------------------------
say "RUNTIME: launch, model load, real chat"
pkill -9 -f "Aria.app/Contents/MacOS" 2>/dev/null || true
# Longer than it looks like it needs to be — confirmed live: a previous
# build's app instance left running (a normal thing to do between builds)
# can still be mid-teardown of its own Metal/GPU state a mere 1s after
# SIGKILL, and starting a fresh MLX process into that window was enough to
# get this run's own sidecar spuriously SIGTERM'd partway through model
# load. 3s gave the GPU handoff room to actually finish before racing it.
sleep 3

# Isolated state dir so the gate never touches real user data; weights are
# shared read-only via symlink (a per-run 6GB copy would be absurd).
GATE_HOME="$(mktemp -d /tmp/aria-gate-XXXXXX)"
HAVE_WEIGHTS=0
if [[ -d "$HOME/.aria/models/gemma-4-12b" ]]; then
  ln -s "$HOME/.aria/models" "$GATE_HOME/models"
  HAVE_WEIGHTS=1
fi

ARIA_HOME="$GATE_HOME" "$APP/Contents/MacOS/aria" >/dev/null 2>"$GATE_HOME/stderr.log" &
SHELL_PID=$!

PORT=""
for _ in $(seq 1 30); do
  sleep 1
  SID_PID="$(pgrep -P "$SHELL_PID" -f aria-sidecar | head -1 || true)"
  if [[ -n "${SID_PID:-}" ]]; then
    PORT="$(lsof -p "$SID_PID" -a -iTCP -sTCP:LISTEN -P 2>/dev/null | awk 'NR==2{sub(/.*:/,"",$9); print $9}')"
    [[ -n "$PORT" ]] && break
  fi
done
if [[ -n "$PORT" ]]; then
  ok "sidecar bound 127.0.0.1:$PORT"
else
  bad "sidecar never bound a port; stderr follows"
  sed 's/^/    /' "$GATE_HOME/stderr.log" || true
fi

if [[ -n "$PORT" && "$HAVE_WEIGHTS" == "1" ]]; then
  # The gate's ARIA_HOME is a fresh dir with no remembered last_model_id, so
  # nothing auto-loads — request the load explicitly (blocks until done).
  curl -s -m 180 -X POST "http://127.0.0.1:$PORT/models/load" \
    -H 'Content-Type: application/json' -d '{"model_id":"gemma-4-12b"}' >/dev/null 2>&1 || true
  LOADED="$(curl -s -m 5 "http://127.0.0.1:$PORT/status" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('loaded'))" 2>/dev/null || true)"
  if [[ "$LOADED" == "True" ]]; then
    ok "model loaded on Metal"
  else
    bad "model failed to load; stderr follows"
    sed 's/^/    /' "$GATE_HOME/stderr.log" || true
  fi

  if [[ "$LOADED" == "True" ]]; then
    REPLY="$(curl -s -m 90 -X POST "http://127.0.0.1:$PORT/chat" \
      -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Reply with the single word: ready"}]}' \
      | python3 -c "import json,sys; print(json.load(sys.stdin).get('content','').strip())" 2>/dev/null || true)"
    if [[ -n "$REPLY" ]]; then
      ok "real GPU chat reply: \"$(printf '%s' "$REPLY" | head -c 60)\""
    else
      bad "chat returned an empty reply"
    fi
  fi
elif [[ -n "$PORT" ]]; then
  warn "gemma-4-12b weights not on this machine — model/chat checks SKIPPED (structure+lifecycle only)"
fi

# --- lifecycle ---------------------------------------------------------------
say "LIFECYCLE: crash recovery + clean quit"
if [[ -n "$PORT" ]]; then
  kill -9 "$(pgrep -P "$SHELL_PID" -f aria-sidecar | head -1)" 2>/dev/null || true
  NEW_SID=""
  for _ in $(seq 1 10); do
    sleep 1
    NEW_SID="$(pgrep -P "$SHELL_PID" -f aria-sidecar | head -1 || true)"
    [[ -n "$NEW_SID" ]] && break
  done
  [[ -n "$NEW_SID" ]] && ok "sidecar auto-restarted after kill -9 (pid $NEW_SID)" \
                      || bad "sidecar did NOT auto-restart after being killed"

  # Quit through the real quit path (same as ⌘Q) — a raw SIGTERM would skip
  # Tauri's exit handlers entirely and orphan the sidecar by definition,
  # failing this check for the wrong reason. Our instance is the only "Aria"
  # running (everything else was killed at gate start), so this targets it.
  osascript -e 'tell application "Aria" to quit' >/dev/null 2>&1 || kill "$SHELL_PID" 2>/dev/null || true
  # Poll instead of one fixed sleep — confirmed live as a source of a flaky
  # false FAIL: right after the auto-restart check above (kill -9, respawn,
  # settle), teardown can occasionally take a beat longer than a flat 3s,
  # and the identical bundle passes cleanly on a slower quit. Only actually
  # fail once nothing has changed for several consecutive checks.
  ORPHANED=1
  for _ in $(seq 1 10); do
    sleep 1
    if ! pgrep -f "Aria.app/Contents/MacOS" >/dev/null 2>&1; then
      ORPHANED=0
      break
    fi
  done
  if [[ "$ORPHANED" == "1" ]]; then
    bad "orphan processes survive quit:"
    pgrep -fl "Aria.app/Contents/MacOS" | sed 's/^/    /'
  else
    ok "quit left zero orphan processes"
  fi
fi

# --- verdict -----------------------------------------------------------------
say "VERDICT"
DMG="$(/usr/bin/find "$DMG_DIR" -name '*.dmg' -maxdepth 1 2>/dev/null | head -1 || true)"
if [[ "$FAIL" -eq 0 ]]; then
  printf '  \033[0;32m✓ RELEASE GATE PASSED (%d checks)\033[0m\n' "$PASS"
  if [[ -n "$DMG" ]]; then
    shasum -a 256 "$DMG" | cut -d' ' -f1 > "$STAMP"
    echo "  stamped $STAMP for $(basename "$DMG")"
  fi
  exit 0
else
  printf '  \033[0;31m✗ RELEASE GATE FAILED (%d passed, %d FAILED) — DO NOT SHIP\033[0m\n' "$PASS" "$FAIL"
  rm -f "$STAMP"
  exit 1
fi
