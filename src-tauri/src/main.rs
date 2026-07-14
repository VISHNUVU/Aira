// Aria — Tauri shell entry point.
//
// Responsibilities of the Rust shell (deliberately thin):
//   1. Spawn the Python sidecar on startup — the PyInstaller-frozen binary in a
//      bundled build, or `python3 app.py` in dev.
//   2. Read the "ARIA_PORT=<n>" line the sidecar prints once it has bound.
//   3. Expose that port to the web UI via the `sidecar_port` command and a
//      `sidecar-ready` event.
//   4. Tear the sidecar down on exit.
//
// All real logic (engine, memory, feedback, trainer, tools) lives in the Python
// sidecar. The web UI talks to it directly over JSON on 127.0.0.1.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use tauri::async_runtime;
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

/// A crash loop (bad build, corrupted model, whatever) must eventually stop
/// retrying rather than hammering the machine forever — 5 attempts is enough
/// to shrug off a transient failure without masking a genuinely broken build.
const MAX_RESTART_ATTEMPTS: u32 = 5;

/// Holds the running sidecar process handle + the port it bound.
#[derive(Default)]
struct Sidecar {
    child: Mutex<Option<CommandChild>>,
    port: Mutex<Option<u16>>,
    restart_attempts: AtomicU32,
}

/// The web UI polls this to discover where the sidecar is listening.
/// Returns `None` until the sidecar has printed its port.
#[tauri::command]
fn sidecar_port(state: State<Sidecar>) -> Option<u16> {
    *state.port.lock().unwrap()
}

/// Tears down the sidecar by PID *and* by path pattern. The onedir build
/// (see aria-sidecar.spec) normally means `child.kill()` alone is enough —
/// unlike the old onefile build, there's no separate extracted worker
/// process to outlive it — but the `pkill -f` sweep stays as a cheap safety
/// net for orphans left behind by anything that bypasses this function
/// entirely (an abnormal kill of the whole app, an old build still cached
/// somewhere, etc.).
fn kill_sidecar(state: &State<Sidecar>) {
    if let Some(child) = state.child.lock().unwrap().take() {
        let _ = child.kill();
    }
    let _ = std::process::Command::new("pkill")
        .args(["-f", "Aria.app/Contents/MacOS/aria-sidecar"])
        .output();
}

/// Called by the web UI right after a successful self-update install: kills
/// this instance's sidecar (the freshly-relaunched new Aria.app has already
/// spawned its own) and exits, the same teardown the window-close handler
/// below does on a normal quit.
#[tauri::command]
fn quit_now(app: tauri::AppHandle) {
    if let Some(state) = app.try_state::<Sidecar>() {
        kill_sidecar(&state);
    }
    app.exit(0);
}

/// PyInstaller's onedir build (see aria-sidecar.spec) needs its `_internal/`
/// dependency tree — but Tauri's externalBin convention only places a
/// single file at Contents/MacOS/aria-sidecar, so `_internal/` ships
/// separately as a bundled Resource (Contents/Resources/_internal) and gets
/// linked into place here, once, on first launch of this exact build.
///
/// Two links, not one — confirmed live by running the frozen binary
/// directly: PyInstaller's macOS bootloader detects it's sitting inside a
/// `.app/Contents/MacOS/` path and, in that mode, looks for the Python
/// runtime specifically at `Contents/Frameworks/Python` (standard macOS
/// bundle convention), not at `_internal/Python` next to itself the way a
/// bare (non-bundled) onedir distribution would. `_internal/Python` is the
/// real dylib PyInstaller collected, so `Contents/Frameworks` just needs to
/// resolve to the same place `_internal` does.
///
/// No-op in dev (no bundled resource on disk) and a no-op on every
/// subsequent launch (the symlinks, or real directories from a previous
/// run, already exist).
fn ensure_sidecar_internal_symlink(app: &tauri::App) {
    let Ok(exe_path) = std::env::current_exe() else { return };
    let Some(exe_dir) = exe_path.parent() else { return }; // Contents/MacOS
    let Some(contents_dir) = exe_dir.parent() else { return }; // Contents

    let Ok(resource_dir) = app.path().resource_dir() else { return };
    let target = resource_dir.join("_internal");
    if !target.exists() {
        return; // dev build / no bundled resource — nothing to link
    }

    for link_path in [exe_dir.join("_internal"), contents_dir.join("Frameworks")] {
        if link_path.exists() {
            continue;
        }
        if let Err(e) = std::os::unix::fs::symlink(&target, &link_path) {
            eprintln!("warning: couldn't link {}: {e}", link_path.display());
        }
    }
}

/// Builds the sidecar command — the frozen onedir binary via Tauri's
/// externalBin convention in a bundled build, or `python3 app.py` straight
/// from source in dev (`tauri dev`, where that binary was never built).
fn build_sidecar_command(app: &AppHandle) -> tauri_plugin_shell::process::Command {
    match app.shell().sidecar("aria-sidecar") {
        Ok(cmd) => cmd.args(["--engine", "auto", "--port", "0"]),
        Err(_) => {
            let python = std::env::var("ARIA_PYTHON").unwrap_or_else(|_| "python3".into());
            let script = std::env::var("ARIA_SIDECAR")
                .unwrap_or_else(|_| "../python-sidecar/app.py".into());
            app.shell()
                .command(python)
                .args([&script, "--engine", "auto", "--port", "0"])
        }
    }
}

/// Spawns the sidecar and pumps its output on a background task. On an
/// unexpected exit (crash, killed by the OS, whatever) — confirmed live as
/// a real, if intermittent, failure mode during heavy model loading — this
/// automatically respawns it rather than leaving the window permanently
/// stuck on "Starting…" with a healthy-looking UI hiding a dead backend.
/// Each respawn re-emits `sidecar-ready` with the new port; the web UI's
/// existing listener (wired once, in boot()) picks that up and self-heals
/// even if it had already confirmed and moved on.
fn spawn_sidecar(app: AppHandle) {
    let cmd = build_sidecar_command(&app);
    let (mut rx, child) = match cmd.spawn() {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("failed to spawn Aria sidecar: {e}");
            return;
        }
    };

    app.state::<Sidecar>().child.lock().unwrap().replace(child);

    async_runtime::spawn(async move {
        let mut exited_unexpectedly = false;
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(bytes) => {
                    let line = String::from_utf8_lossy(&bytes);
                    for l in line.lines() {
                        if let Some(rest) = l.trim().strip_prefix("ARIA_PORT=") {
                            if let Ok(port) = rest.parse::<u16>() {
                                eprintln!("sidecar listening on 127.0.0.1:{port}");
                                *app.state::<Sidecar>().port.lock().unwrap() = Some(port);
                                app.state::<Sidecar>()
                                    .restart_attempts
                                    .store(0, Ordering::SeqCst);
                                let _ = app.emit("sidecar-ready", port);
                            }
                        }
                    }
                }
                CommandEvent::Stderr(bytes) => {
                    // Was previously dropped entirely — every Python-side
                    // traceback and error print (including auto_load_last_model's
                    // own exception handler) vanished with no trace, which is
                    // exactly what made a real, printed error look like a
                    // silent hang from the outside. Prefixed so it's easy to
                    // tell apart from this shell's own log lines.
                    let line = String::from_utf8_lossy(&bytes);
                    for l in line.lines() {
                        if !l.trim().is_empty() {
                            eprintln!("[sidecar] {l}");
                        }
                    }
                }
                CommandEvent::Terminated(payload) => {
                    eprintln!("sidecar exited unexpectedly: {payload:?}");
                    exited_unexpectedly = true;
                }
                _ => {}
            }
        }

        if !exited_unexpectedly {
            return; // stdout closed because we killed it deliberately (quit/restart)
        }
        *app.state::<Sidecar>().port.lock().unwrap() = None;

        let attempts = app
            .state::<Sidecar>()
            .restart_attempts
            .fetch_add(1, Ordering::SeqCst)
            + 1;
        if attempts > MAX_RESTART_ATTEMPTS {
            eprintln!("sidecar crashed {attempts} times in a row — giving up on auto-restart");
            return;
        }
        eprintln!("restarting sidecar (attempt {attempts}/{MAX_RESTART_ATTEMPTS})…");
        tokio::time::sleep(Duration::from_millis(500)).await;
        spawn_sidecar(app);
    });
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(Sidecar::default())
        .invoke_handler(tauri::generate_handler![sidecar_port, quit_now])
        .setup(|app| {
            // Reap any sidecar orphaned by a crash or force-quit of a previous
            // launch — normal quit already tears it down via the window
            // Destroyed handler below, but that never runs if the app was
            // killed abnormally. A stray old sidecar left running would keep
            // answering on its own port with stale in-memory state (e.g. an
            // old APP_VERSION), which is exactly what makes an update look
            // "stuck" even after the new build is actually installed.
            let _ = std::process::Command::new("pkill")
                .args(["-f", "Aria.app/Contents/MacOS/aria-sidecar"])
                .output();

            ensure_sidecar_internal_symlink(app);
            spawn_sidecar(app.handle().clone());

            Ok(())
        })
        .on_window_event(|window, event| {
            // Kill the sidecar when the main window closes. Covers the
            // red-button close path; ⌘Q / Dock "Quit" instead surface as
            // RunEvent::ExitRequested below, which this handler never sees.
            if let tauri::WindowEvent::Destroyed = event {
                if let Some(state) = window.app_handle().try_state::<Sidecar>() {
                    kill_sidecar(&state);
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building Aria")
        .run(|app_handle, event| {
            // ⌘Q, Dock "Quit", and `osascript ... to quit` all terminate the
            // app through this event rather than a window Destroyed event —
            // confirmed live: quitting this way left the sidecar running as
            // an orphan (reparented to launchd) with no crash report, since
            // nothing killed it before the process exited. Covering both
            // ExitRequested and Exit means whichever fires first tears the
            // sidecar down; .take() makes the second one a no-op.
            match event {
                tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit => {
                    if let Some(state) = app_handle.try_state::<Sidecar>() {
                        kill_sidecar(&state);
                    }
                }
                _ => {}
            }
        });
}
