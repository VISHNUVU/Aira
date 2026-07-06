# Distributing Aria

How to turn the Aria source tree into an installable macOS app that other
people can run — first unsigned (free), then optionally signed + notarized so
it opens with no Gatekeeper warning.

> **Who can run Aria:** Apple Silicon Macs (M1 or newer). The engine is built
> on Apple's MLX framework, which is Apple-Silicon-only. Intel Macs are not
> supported.

---

## 1. One-command build (unsigned)

On an Apple Silicon Mac with the build tools installed:

```bash
scripts/build-macos.sh
```

That single command:

1. Freezes the Python sidecar into a standalone binary (PyInstaller) — end
   users need **no Python installed**.
2. Stages it where Tauri expects (`src-tauri/binaries/aria-sidecar-aarch64-apple-darwin`).
3. Generates the app icon set if missing.
4. Installs the JS deps and compiles the Rust/Tauri shell.
5. Produces **`Aria.app`** and **`Aria_0.1.0_aarch64.dmg`** under
   `src-tauri/target/aarch64-apple-darwin/release/bundle/`.

### Build prerequisites

Install these on the **build** machine (not needed by end users):

| Tool | Install |
|------|---------|
| Xcode Command Line Tools | `xcode-select --install` |
| Rust + Cargo | https://rustup.rs |
| Node.js + npm | https://nodejs.org  or  `brew install node` |
| Python 3.11+ | system `python3` is fine |

Tauri CLI and PyInstaller are installed automatically by the script.

---

## 2. What end users do

Send them the `.dmg`. They:

1. Open the `.dmg`, drag **Aria** to Applications.
2. **First launch only:** right-click (or Control-click) Aria → **Open** →
   **Open** again. This is the one-time Gatekeeper step for unsigned apps.
   After that it opens normally.
3. On first run Aria downloads its model (~7 GB) into `~/.aria`. This is
   one-time and requires an internet connection; everything after is offline.

If you'd rather users never see the right-click step, sign + notarize (below).

---

## 3. Signed + notarized build (optional, removes the warning)

This makes Aria open with a normal double-click on any Mac — no right-click,
no warning. It requires a paid **Apple Developer account** ($99/yr) and a
**Developer ID Application** certificate.

### One-time setup

1. Join the Apple Developer Program and, in Xcode or the developer portal,
   create a **Developer ID Application** certificate. Confirm it's installed:
   ```bash
   security find-identity -v -p codesigning
   # → "Developer ID Application: Your Name (TEAMID1234)"
   ```
2. Create an **app-specific password** at
   https://appleid.apple.com → Sign-In & Security → App-Specific Passwords.

### Build, then sign

```bash
# 1) produce the unsigned app + dmg
scripts/build-macos.sh

# 2) sign, notarize, staple  (credentials via env — never hardcoded)
export SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID1234)"
export APPLE_ID="you@example.com"
export APPLE_TEAM_ID="TEAMID1234"
export APPLE_PASSWORD="abcd-efgh-ijkl-mnop"   # the app-specific password
scripts/sign-and-notarize.sh
```

The script deep-signs the bundle (including the embedded sidecar) with the
hardened runtime, uploads to Apple for notarization, waits for the result, and
staples the ticket to the `.dmg`. When it finishes, the `.dmg` opens cleanly on
any Mac.

> **Tip — keychain profile:** instead of the three `APPLE_*` vars you can
> pre-store credentials once with
> `xcrun notarytool store-credentials` and then set `NOTARY_PROFILE=<name>`.

### Why the entitlements?

PyInstaller's frozen binary uses JIT-adjacent memory and loads bundled dylibs
at runtime, so notarization under the hardened runtime needs a small
entitlements set (`allow-jit`, `allow-unsigned-executable-memory`,
`disable-library-validation`, `allow-dyld-environment-variables`). The script
writes `src-tauri/entitlements.plist` with exactly these if it's missing.

---

## 4. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `cargo: command not found` | Install Rust: https://rustup.rs, then re-open the shell. |
| `npm: command not found` | Install Node: https://nodejs.org. |
| PyInstaller cache error | The script pins its cache to `python-sidecar/.pyi-cache`; delete it and retry. |
| "Aria is damaged" on a user's Mac | The app was moved before notarization stapled, or it's unsigned and quarantined — use right-click → Open, or ship the notarized build. |
| App opens but chat says no model | Expected on first run — click **Open Models →** and let the ~7 GB download finish. |
| Notarization rejected | Run `xcrun notarytool log <submission-id> --keychain-profile <p>` to see why; usually a missing entitlement or an unsigned nested binary. |

---

## 5. What ships inside the app

```
Aria.app/Contents/
├── MacOS/Aria                        ← Rust/Tauri shell (tiny)
├── Resources/                        ← web UI (HTML/CSS/JS) + icons
└── Resources/ (or MacOS/) aria-sidecar-aarch64-apple-darwin
                                      ← frozen Python sidecar (~20 MB, all
                                        engine/memory/training/voice logic)
```

The model weights are **not** bundled — they download on first launch, which
keeps the app small and lets you update the model independently.
