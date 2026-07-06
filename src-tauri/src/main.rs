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

use std::sync::Mutex;

use tauri::async_runtime;
use tauri::{Emitter, Manager, State};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

/// Holds the running sidecar process handle + the port it bound.
#[derive(Default)]
struct Sidecar {
    child: Mutex<Option<CommandChild>>,
    port: Mutex<Option<u16>>,
}

/// The web UI polls this to discover where the sidecar is listening.
/// Returns `None` until the sidecar has printed its port.
#[tauri::command]
fn sidecar_port(state: State<Sidecar>) -> Option<u16> {
    *state.port.lock().unwrap()
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(Sidecar::default())
        .invoke_handler(tauri::generate_handler![sidecar_port])
        .setup(|app| {
            let handle = app.handle().clone();

            // In a bundled build the frozen sidecar is shipped as a Tauri
            // "sidecar" binary (see externalBin in tauri.conf.json). In dev
            // (`tauri dev`) that binary doesn't exist, so we fall back to
            // running the Python source directly.
            let sidecar_cmd = app.shell().sidecar("aria-sidecar");

            let cmd = match sidecar_cmd {
                Ok(cmd) => cmd.args(["--engine", "auto", "--port", "0"]),
                Err(_) => {
                    // Dev fallback: python3 ../python-sidecar/app.py --port 0
                    let python =
                        std::env::var("ARIA_PYTHON").unwrap_or_else(|_| "python3".into());
                    let script = std::env::var("ARIA_SIDECAR")
                        .unwrap_or_else(|_| "../python-sidecar/app.py".into());
                    app.shell()
                        .command(python)
                        .args([&script, "--engine", "auto", "--port", "0"])
                }
            };

            let (mut rx, child) = cmd.spawn().expect("failed to spawn Aria sidecar");

            // Stash the child so we can kill it on window close.
            app.state::<Sidecar>()
                .child
                .lock()
                .unwrap()
                .replace(child);

            // Pump the sidecar's stdout/stderr on a background task. The first
            // "ARIA_PORT=<n>" line resolves the port and fires `sidecar-ready`.
            async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    if let CommandEvent::Stdout(bytes) = event {
                        let line = String::from_utf8_lossy(&bytes);
                        for l in line.lines() {
                            if let Some(rest) = l.trim().strip_prefix("ARIA_PORT=") {
                                if let Ok(port) = rest.parse::<u16>() {
                                    eprintln!("sidecar listening on 127.0.0.1:{port}");
                                    *handle.state::<Sidecar>().port.lock().unwrap() =
                                        Some(port);
                                    let _ = handle.emit("sidecar-ready", port);
                                }
                            }
                        }
                    }
                }
            });

            Ok(())
        })
        .on_window_event(|window, event| {
            // Kill the sidecar when the main window closes.
            if let tauri::WindowEvent::Destroyed = event {
                if let Some(state) = window.app_handle().try_state::<Sidecar>() {
                    if let Some(child) = state.child.lock().unwrap().take() {
                        let _ = child.kill();
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running Aria");
}
