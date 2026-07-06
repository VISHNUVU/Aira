#!/usr/bin/env bash
# ============================================================================
# sign-and-notarize.sh — OPTIONAL: ship Aria with no Gatekeeper warning
# ============================================================================
# The default build (build-macos.sh) is UNSIGNED — anyone can run it via
# right-click → Open. To distribute so it opens with a normal double-click on
# any Mac, the .app must be code-signed with an Apple "Developer ID Application"
# certificate AND notarized by Apple. That requires a paid Apple Developer
# account ($99/yr). This script does both, then staples the ticket.
#
# ── What you need (all supplied via ENVIRONMENT VARIABLES — never hardcoded) ──
#   SIGN_IDENTITY   Full name of your Developer ID Application cert, e.g.
#                   "Developer ID Application: Jane Doe (TEAMID1234)".
#                   List yours with:  security find-identity -v -p codesigning
#   APPLE_ID        Your Apple Developer account email (for notarization).
#   APPLE_TEAM_ID   Your 10-char Team ID (e.g. TEAMID1234).
#   APPLE_PASSWORD  An app-specific password (appleid.apple.com → Sign-In &
#                   Security → App-Specific Passwords). NOT your Apple ID
#                   password. Alternatively pre-store a notarytool keychain
#                   profile and set NOTARY_PROFILE instead of the three vars.
#
# ── Usage ──
#   export SIGN_IDENTITY="Developer ID Application: Jane Doe (TEAMID1234)"
#   export APPLE_ID="jane@example.com"
#   export APPLE_TEAM_ID="TEAMID1234"
#   export APPLE_PASSWORD="abcd-efgh-ijkl-mnop"
#   scripts/sign-and-notarize.sh [path/to/Aria.app] [path/to/Aria.dmg]
#
# If paths are omitted, the script finds the most recent build under
# src-tauri/target/*/release/bundle/.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

say()  { printf '\n\033[1;34m▸ %s\033[0m\n' "$*"; }
ok()   { printf '  \033[0;32m✓ %s\033[0m\n' "$*"; }
die()  { printf '\n\033[0;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

ENTITLEMENTS="$ROOT/src-tauri/entitlements.plist"
BUNDLE_GLOB="$ROOT/src-tauri/target"

# --- 0. preflight -----------------------------------------------------------
say "Preflight"
[[ "$(uname -s)" == "Darwin" ]] || die "macOS required."
command -v codesign  >/dev/null || die "codesign not found (install Xcode CLT)."
command -v xcrun     >/dev/null || die "xcrun not found (install Xcode CLT)."
: "${SIGN_IDENTITY:?set SIGN_IDENTITY (see header)}"

USE_PROFILE="${NOTARY_PROFILE:-}"
if [[ -z "$USE_PROFILE" ]]; then
  : "${APPLE_ID:?set APPLE_ID or NOTARY_PROFILE}"
  : "${APPLE_TEAM_ID:?set APPLE_TEAM_ID or NOTARY_PROFILE}"
  : "${APPLE_PASSWORD:?set APPLE_PASSWORD (app-specific) or NOTARY_PROFILE}"
fi
ok "signing identity + notarization credentials present"

# --- 1. resolve app + dmg paths --------------------------------------------
APP_PATH="${1:-}"
DMG_PATH="${2:-}"
if [[ -z "$APP_PATH" ]]; then
  APP_PATH="$(/usr/bin/find "$BUNDLE_GLOB" -name 'Aria.app' 2>/dev/null | head -1 || true)"
fi
[[ -n "$APP_PATH" && -d "$APP_PATH" ]] || die "Aria.app not found — run scripts/build-macos.sh first."
if [[ -z "$DMG_PATH" ]]; then
  DMG_PATH="$(/usr/bin/find "$BUNDLE_GLOB" -name '*.dmg' 2>/dev/null | head -1 || true)"
fi
ok "app: $APP_PATH"
[[ -n "$DMG_PATH" ]] && ok "dmg: $DMG_PATH"

# --- 2. entitlements (hardened runtime needs these for a frozen sidecar) ----
# PyInstaller's bootloader uses JIT-adjacent memory + loads unsigned dylibs at
# runtime, so the hardened runtime must allow those. This is the minimal set.
if [[ ! -f "$ENTITLEMENTS" ]]; then
  say "Writing entitlements.plist"
  cat > "$ENTITLEMENTS" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>com.apple.security.cs.allow-jit</key><true/>
  <key>com.apple.security.cs.allow-unsigned-executable-memory</key><true/>
  <key>com.apple.security.cs.disable-library-validation</key><true/>
  <key>com.apple.security.cs.allow-dyld-environment-variables</key><true/>
</dict>
</plist>
PLIST
  ok "wrote $ENTITLEMENTS"
fi

# --- 3. deep-sign the app ---------------------------------------------------
# Sign inside-out: the embedded sidecar binary first, then the whole bundle.
say "Code-signing (hardened runtime)"
SIDECAR_IN_APP="$(/usr/bin/find "$APP_PATH/Contents" -name 'aria-sidecar*' -type f 2>/dev/null | head -1 || true)"
if [[ -n "$SIDECAR_IN_APP" ]]; then
  codesign --force --timestamp --options runtime \
    --entitlements "$ENTITLEMENTS" \
    --sign "$SIGN_IDENTITY" "$SIDECAR_IN_APP"
  ok "signed embedded sidecar"
fi
codesign --force --deep --timestamp --options runtime \
  --entitlements "$ENTITLEMENTS" \
  --sign "$SIGN_IDENTITY" "$APP_PATH"
codesign --verify --deep --strict --verbose=2 "$APP_PATH"
ok "app signed + verified"

# --- 4. notarize ------------------------------------------------------------
# Notarize the .dmg if we have one (preferred — it's what users download),
# else zip the .app and notarize that.
say "Notarizing (uploading to Apple — can take a few minutes)"
NOTARIZE_TARGET="$DMG_PATH"
CLEANUP_ZIP=""
if [[ -z "$NOTARIZE_TARGET" ]]; then
  NOTARIZE_TARGET="$ROOT/Aria-notarize.zip"
  /usr/bin/ditto -c -k --keepParent "$APP_PATH" "$NOTARIZE_TARGET"
  CLEANUP_ZIP="$NOTARIZE_TARGET"
fi

if [[ -n "$USE_PROFILE" ]]; then
  xcrun notarytool submit "$NOTARIZE_TARGET" \
    --keychain-profile "$USE_PROFILE" --wait
else
  xcrun notarytool submit "$NOTARIZE_TARGET" \
    --apple-id "$APPLE_ID" --team-id "$APPLE_TEAM_ID" \
    --password "$APPLE_PASSWORD" --wait
fi
ok "notarization accepted"

# --- 5. staple --------------------------------------------------------------
say "Stapling ticket"
if [[ -n "$DMG_PATH" ]]; then
  xcrun stapler staple "$DMG_PATH"
  xcrun stapler staple "$APP_PATH" || true
  ok "stapled dmg + app"
else
  xcrun stapler staple "$APP_PATH"
  ok "stapled app"
fi
[[ -n "$CLEANUP_ZIP" ]] && rm -f "$CLEANUP_ZIP"

# --- 6. final gatekeeper assessment ----------------------------------------
say "Gatekeeper assessment"
spctl --assess --type execute --verbose=4 "$APP_PATH" 2>&1 | sed 's/^/  /' || true
cat <<EOF

  ✓ Signed, notarized, stapled. Distribute the .dmg — it now opens with a
    normal double-click on any Mac, no right-click and no warning.
EOF
