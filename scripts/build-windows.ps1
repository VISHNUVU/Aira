# ============================================================================
# build-windows.ps1 — unsigned installable Aria .msi for Windows x64
# ============================================================================
# Mirrors scripts/build-macos.sh's structure for the Windows target:
#   1. Freezes the Python sidecar into a standalone binary (PyInstaller)
#   2. Stages it at the Tauri externalBin target-triple path
#   3. Compiles the Tauri (Rust) shell and bundles an .msi
#   4. Runs a lightweight boot smoke check (see verify-build.ps1's macOS
#      counterpart, scripts/verify-build.sh, for why this can't be the same
#      depth here: no real GPU/model weights are downloaded in this build,
#      and llama.cpp's inference path is proven separately and already —
#      see python-sidecar/tests/test_llamacpp_driver.py's real-GGUF smoke
#      test, run on macOS's CPU backend since llama-cpp-python itself is
#      cross-platform).
#
# The result is UNSIGNED — no code-signing certificate exists yet, so
# Windows SmartScreen shows a one-time "unknown publisher" warning on first
# launch, the same posture as today's unsigned macOS .dmg.
#
# Requirements on the BUILD machine (or CI runner — see
# .github/workflows/build-windows.yml, the machine this is actually meant
# to run on today):
#   - Rust + Cargo               (https://rustup.rs)
#   - Node.js + npm              (https://nodejs.org)
#   - Python 3.11+
#   - Visual Studio Build Tools (MSVC + the Windows SDK's llvm-rc, needed to
#     embed the .exe's icon/version resource — confirmed missing is the
#     actual wall a macOS host hits; see this script's design notes)
# ============================================================================
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Triple = "x86_64-pc-windows-msvc"
$SidecarDir = Join-Path $Root "python-sidecar"
$BinDir = Join-Path $Root "src-tauri\binaries"
$StagedBin = Join-Path $BinDir "aria-sidecar-$Triple.exe"

function Say($msg)  { Write-Host "`n▸ $msg" -ForegroundColor Blue }
function Ok($msg)   { Write-Host "  ✓ $msg" -ForegroundColor Green }
function Die($msg)  { Write-Host "`n✗ $msg" -ForegroundColor Red; exit 1 }

# --- 0. sanity: platform + toolchain ---------------------------------------
Say "Checking build host"
if (-not $IsWindows) { Die "Windows required (this builds a .msi)." }
if (-not (Get-Command python -ErrorAction SilentlyContinue)) { Die "python not found." }
if (-not (Get-Command cargo -ErrorAction SilentlyContinue))  { Die "Rust/Cargo not found. Install: https://rustup.rs" }
if (-not (Get-Command npm -ErrorAction SilentlyContinue))    { Die "Node/npm not found. Install: https://nodejs.org" }
Ok "Windows, python + cargo + npm present"

# --- 1. freeze the Python sidecar -------------------------------------------
Say "Freezing Python sidecar (PyInstaller)"
Push-Location $SidecarDir
try {
    if (-not (Test-Path ".venv")) {
        Write-Host "  creating build venv (.venv)"
        python -m venv .venv
    }
    & ".venv\Scripts\Activate.ps1"
    python -m pip install --upgrade pip | Out-Null

    # requirements.lock is a pip freeze of the macOS build venv — it pins
    # mlx/mlx-lm/mlx-embeddings/kokoro/mflux, none of which have Windows
    # wheels at all (mlx is Apple-Silicon-only), so it can never be reused
    # here. Confirmed live: a first CI run failed immediately on
    # `mlx==0.31.2` with "No matching distribution found". Always use the
    # unpinned extras path on this platform instead — a genuine
    # requirements-windows.lock (generated from a verified Windows install)
    # is future work once this pipeline itself has run successfully once.
    Write-Host "  installing dependencies (unpinned — no Windows lockfile yet)"
    pip install -e ".[llamacpp,memory]" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  full backend install failed — building lean sidecar (fake engine only)"
        pip install -e ".[dev]" 2>$null
        if ($LASTEXITCODE -ne 0) { pip install -e "." }
    }
    pip install "pyinstaller>=6" | Out-Null
    Ok "PyInstaller ready"

    $env:PYINSTALLER_CONFIG_DIR = Join-Path $SidecarDir ".pyi-cache"
    New-Item -ItemType Directory -Force -Path $env:PYINSTALLER_CONFIG_DIR | Out-Null

    Write-Host "  running PyInstaller (aria-sidecar.spec)…"
    pyinstaller aria-sidecar.spec --noconfirm --clean | Out-Null
    if (-not (Test-Path "dist\aria-sidecar\aria-sidecar.exe")) {
        Die "PyInstaller did not produce dist\aria-sidecar\aria-sidecar.exe (onedir)"
    }
    $size = (Get-ChildItem "dist\aria-sidecar" -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB
    Ok ("sidecar frozen (onedir): {0:N0} MB total" -f $size)
    deactivate
} finally {
    Pop-Location
}

# --- 2. stage the onedir build for Tauri ------------------------------------
# Unlike macOS (see build-macos.sh's step 5b), Windows' PyInstaller output
# has no framework/symlink structure to preserve, so no runtime symlink/copy
# dance is needed (contrast main.rs's ensure_sidecar_internal_symlink,
# macOS-only). Instead, tauri.windows.conf.json declares `_internal` as a
# bundle resource — confirmed from Tauri's own source
# (tauri-utils::platform::resource_dir_from) that resource_dir() always
# equals the executable's own directory on Windows, so this lands
# `_internal` directly next to the sidecar exe at build time, exactly where
# PyInstaller's onedir bootloader looks for it.
Say "Staging sidecar for Tauri (onedir: externalBin + resources)"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
Copy-Item (Join-Path $SidecarDir "dist\aria-sidecar\aria-sidecar.exe") $StagedBin -Force
Ok "staged executable: $StagedBin"

$SidecarInternalDir = Join-Path $Root "src-tauri\resources\sidecar-internal"
Remove-Item -Recurse -Force $SidecarInternalDir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $SidecarInternalDir | Out-Null
Copy-Item (Join-Path $SidecarDir "dist\aria-sidecar\_internal") (Join-Path $SidecarInternalDir "_internal") -Recurse
Ok "staged dependency tree: $SidecarInternalDir\_internal"

# --- 3. JS deps + Tauri CLI --------------------------------------------------
Say "Installing JS dependencies"
Push-Location $Root
try {
    npm install | Out-Null
    Ok "node_modules ready"

    # --- 4. compile the app + bundle -----------------------------------------
    Say "Building Aria.exe (tauri build — first Rust compile is slow)"
    Remove-Item -Recurse -Force "src-tauri\target\$Triple\release\bundle\msi" -ErrorAction SilentlyContinue
    npx tauri build --target $Triple --bundles msi
    if ($LASTEXITCODE -ne 0) { Die "tauri build failed" }
} finally {
    Pop-Location
}

# --- 5. lightweight boot smoke check -----------------------------------------
# Not the same depth as macOS's release gate (no real model weights are
# downloaded here, and llama.cpp's actual inference path is already proven
# separately — see python-sidecar/tests/test_llamacpp_driver.py's real-GGUF
# smoke test). This only proves the frozen sidecar boots, binds a port, and
# answers /status without crashing — the meaningful bar for a first,
# unsigned Windows build.
#
# Deliberately does NOT run $StagedBin (the pre-build staging copy in
# src-tauri\binaries\) — that copy has no _internal/ next to it, since
# PyInstaller's dependency tree only gets placed alongside the executable
# via Tauri's own Windows resource resolution (tauri.windows.conf.json maps
# it there — see that file's comment for why: Tauri's resource_dir() on
# Windows always equals the exe's own directory, confirmed from
# tauri-utils' resource_dir_from()). So this runs the actual build OUTPUT
# instead, where Tauri really placed the sidecar exe + resources together.
Say "Boot smoke check"
$BuiltExe = Get-ChildItem "src-tauri\target\$Triple\release" -Filter "aria-sidecar-*.exe" -File |
    Select-Object -First 1
if (-not $BuiltExe) { Die "no built sidecar exe found under src-tauri\target\$Triple\release" }
if (-not (Test-Path (Join-Path $BuiltExe.Directory "_internal"))) {
    Die "_internal not found next to $($BuiltExe.FullName) — tauri.windows.conf.json's resources mapping didn't land where PyInstaller's bootloader expects it"
}
Ok "found built sidecar with _internal alongside: $($BuiltExe.FullName)"
$proc = Start-Process -FilePath $BuiltExe.FullName -ArgumentList "--engine", "auto", "--port", "0" -PassThru `
    -RedirectStandardOutput "sidecar-boot.log" -RedirectStandardError "sidecar-boot-err.log"
Start-Sleep -Seconds 8
if ($proc.HasExited) {
    Write-Host "--- sidecar-boot.log ---"
    Get-Content "sidecar-boot.log" -ErrorAction SilentlyContinue | Write-Host
    Write-Host "--- sidecar-boot-err.log ---"
    Get-Content "sidecar-boot-err.log" -ErrorAction SilentlyContinue | Write-Host
    Die "sidecar exited immediately (exit code $($proc.ExitCode)) — logs above"
}
$portLine = Get-Content "sidecar-boot.log" | Select-String "ARIA_PORT=(\d+)"
if (-not $portLine) {
    Stop-Process -Id $proc.Id -Force
    Write-Host "--- sidecar-boot-err.log ---"
    Get-Content "sidecar-boot-err.log" -ErrorAction SilentlyContinue | Write-Host
    Die "sidecar never printed ARIA_PORT= — logs above"
}
$port = $portLine.Matches[0].Groups[1].Value
try {
    $status = Invoke-RestMethod -Uri "http://127.0.0.1:$port/status" -TimeoutSec 5
    Ok "sidecar answered /status: engine=$($status.engine)"
} finally {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
}

# --- 6. report artifact ------------------------------------------------------
$BundleDir = Join-Path $Root "src-tauri\target\$Triple\release\bundle\msi"
Say "Build complete"
$Msi = Get-ChildItem $BundleDir -Filter "*.msi" | Select-Object -First 1
if ($Msi) {
    Ok "MSI: $($Msi.FullName)"
    Write-Host "`n  Next steps:"
    Write-Host "    • Distribute: share the .msi. Unsigned — first launch shows a one-time"
    Write-Host "                  SmartScreen 'unknown publisher' prompt (Run anyway)."
    Write-Host "    • On first launch Aria downloads its model — one-time, several minutes."
} else {
    Die "no .msi found under $BundleDir"
}
