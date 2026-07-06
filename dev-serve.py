#!/usr/bin/env python3
"""Aria dev server — serve the src/ frontend for browser-based development.

This is ONLY for iterating on the UI without building the Tauri shell. It serves
the static files in src/ on http://localhost:1420. In this mode there is no Rust
shell to spawn the sidecar, so start the sidecar yourself on the fixed dev port:

    # terminal 1 — the sidecar (fake engine needs no model/weights)
    python python-sidecar/app.py --engine fake --port 8765

    # terminal 2 — the UI
    npm run dev        # -> this script -> http://localhost:1420

The frontend falls back to http://127.0.0.1:8765 when it isn't running inside
Tauri, so the two line up automatically.

For the real desktop app (bundled sidecar, dynamic port, native window) use
`npm run tauri:dev` instead — that path does not use this script.
"""
from __future__ import annotations

import http.server
import os
import socketserver

PORT = int(os.environ.get("ARIA_DEV_PORT", "1420"))
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        # No caching in dev so edits show up on reload.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} - {fmt % args}")


def main() -> None:
    os.chdir(ROOT)
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"▸ Aria UI (dev) on http://localhost:{PORT}")
        print(f"  serving {ROOT}")
        print("  make sure the sidecar is running:  "
              "python python-sidecar/app.py --engine fake --port 8765")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n▸ stopped")


if __name__ == "__main__":
    main()
