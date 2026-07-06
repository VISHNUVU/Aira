#!/usr/bin/env bash
# ============================================================================
# publish-update.sh — push a built .dmg to the self-hosted update server
# ============================================================================
# Run this AFTER scripts/build-macos.sh has produced a fresh .dmg. It:
#   1. Computes the .dmg's sha256 (checked by the app before it'll offer to
#      let you open a downloaded update — see update_checker.py)
#   2. Writes a versioned + a "latest.json" manifest
#   3. Uploads both the manifest and the .dmg to the VPS over the dedicated
#      deploy key set up for this (~/.ssh/aria_deploy)
#
# The version comes from src-tauri/tauri.conf.json — bump that (and
# python-sidecar/app.py's APP_VERSION, kept in lockstep by hand) before
# running this, or the manifest will just re-publish the same version.
#
# Usage:
#   scripts/publish-update.sh "Release notes go here"
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

VPS_HOST="76.13.244.25"
VPS_KEY="$HOME/.ssh/aria_deploy"
REMOTE_DIR="/opt/aria-updates/releases"
NOTES="${1:-No release notes provided.}"

VERSION="$(python3 -c "import json; print(json.load(open('src-tauri/tauri.conf.json'))['version'])")"
DMG_PATH="$(find "src-tauri/target/aarch64-apple-darwin/release/bundle/dmg" -name '*.dmg' -maxdepth 1 | head -1)"
[[ -f "$DMG_PATH" ]] || { echo "✗ no .dmg found — run scripts/build-macos.sh first" >&2; exit 1; }

ASSET_NAME="Aria_${VERSION}_aarch64.dmg"
SHA256="$(shasum -a 256 "$DMG_PATH" | cut -d' ' -f1)"
SIZE="$(stat -f%z "$DMG_PATH")"
PUBLISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ASSET_URL="http://${VPS_HOST}:8099/releases/${ASSET_NAME}"

echo "▸ Publishing Aria v${VERSION}"
echo "  dmg:    $DMG_PATH ($SIZE bytes)"
echo "  sha256: $SHA256"

MANIFEST_FILE="$(mktemp)"
python3 - "$VERSION" "$NOTES" "$PUBLISHED_AT" "$ASSET_NAME" "$ASSET_URL" "$SHA256" "$SIZE" > "$MANIFEST_FILE" <<'PY'
import json, sys
version, notes, published_at, asset_name, asset_url, sha256, size = sys.argv[1:8]
print(json.dumps({
    "version": version,
    "notes": notes,
    "published_at": published_at,
    "asset_name": asset_name,
    "asset_url": asset_url,
    "sha256": sha256,
    "size": int(size),
}, indent=2))
PY

echo "▸ Uploading .dmg (this can take a while over the VPS's uplink)"
scp -i "$VPS_KEY" -o StrictHostKeyChecking=accept-new \
  "$DMG_PATH" "root@${VPS_HOST}:${REMOTE_DIR}/${ASSET_NAME}"

echo "▸ Publishing manifest"
scp -i "$VPS_KEY" -o StrictHostKeyChecking=accept-new \
  "$MANIFEST_FILE" "root@${VPS_HOST}:${REMOTE_DIR}/latest.json"
rm -f "$MANIFEST_FILE"

# scp preserves the local file's mode — mktemp's manifest comes out 600,
# which nginx's worker (running as an unprivileged user, not root) can't
# read, silently 403ing every check. World-readable is fine here; nothing
# in the manifest or the .dmg is a secret.
ssh -i "$VPS_KEY" -o StrictHostKeyChecking=accept-new "root@${VPS_HOST}" \
  "chmod 644 '${REMOTE_DIR}/${ASSET_NAME}' '${REMOTE_DIR}/latest.json'"

echo "✓ Published — verify: curl ${ASSET_URL%/*}/latest.json"
