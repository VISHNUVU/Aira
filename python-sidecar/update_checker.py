"""Update checking against a private GitHub repo's Releases.

Design: "check + download, manual install" — not a silent auto-updater.
GitHub's REST API requires two authenticated round-trips for a private
repo's release assets (list releases -> resolve one asset's `url` -> fetch
that url with an octet-stream Accept header), which the standard
`tauri-plugin-updater` static-JSON-endpoint fetcher can't do in one shot.
Rather than build a full custom silent-install flow (replace-the-running-app
logic, signature verification, relaunch), this does the two authenticated
calls itself and hands the user a downloaded .dmg to open — same manual
last step as today, just triggered from inside the app instead of the user
checking GitHub by hand.

The token is a fine-grained, single-repo, read-only ("Contents") GitHub PAT
— never the broad OAuth token a `gh auth login` session holds. It's read
from a bundled `.update_token` file (see aria-sidecar.spec), not committed
to git. No file means update checks are silently disabled — this is an
optional feature, not something that should ever block the app.
"""
from __future__ import annotations

import json
import os
import ssl
import sys
import urllib.error
import urllib.request

REPO = "VISHNUVU/Aira"
API_BASE = f"https://api.github.com/repos/{REPO}"


def _ssl_context() -> ssl.SSLContext:
    """Explicit certifi-backed context instead of relying on the system's
    default cert store — python.org/framework macOS builds commonly ship
    without their OpenSSL cert path populated (the "Install Certificates
    .command" step), which makes urlopen fail with CERTIFICATE_VERIFY_FAILED
    even though the machine's own trust store is fine. certifi is already a
    transitive dependency here (via huggingface_hub), so this costs nothing
    extra to bundle."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _bundle_dir() -> str:
    """Resolve the token file's directory whether running from source
    (python-sidecar/) or a frozen PyInstaller binary (sys._MEIPASS)."""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _load_token() -> str | None:
    path = os.path.join(_bundle_dir(), ".update_token")
    try:
        with open(path) as f:
            token = f.read().strip()
            return token or None
    except OSError:
        return None


def _api_get(url: str, token: str, accept: str = "application/vnd.github+json") -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": accept,
        "X-GitHub-Api-Version": "2022-11-28",
    })
    try:
        with urllib.request.urlopen(req, timeout=15, context=_ssl_context()) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _parse_version(tag: str) -> tuple:
    """'v1.2.3' or '1.2.3' -> (1, 2, 3), for a simple numeric comparison."""
    cleaned = tag.lstrip("vV")
    parts = []
    for p in cleaned.split("."):
        digits = "".join(c for c in p if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


class UpdateChecker:
    def __init__(self, current_version: str):
        self.current_version = current_version
        self._token = _load_token()

    @property
    def available(self) -> bool:
        return self._token is not None

    def check(self) -> dict:
        """Look up the latest GitHub release and compare to current_version."""
        if not self._token:
            return {"enabled": False, "error": "update checks not configured for this build"}

        try:
            status, body = _api_get(f"{API_BASE}/releases/latest", self._token)
        except urllib.error.URLError as e:
            # Includes SSL failures and "offline" (this app is meant to work
            # without a network connection) — never let an update check
            # crash the request, just report it couldn't reach GitHub.
            return {"enabled": True, "ok": False, "error": f"couldn't reach GitHub: {e.reason}"}
        if status != 200:
            try:
                msg = json.loads(body).get("message", f"HTTP {status}")
            except (json.JSONDecodeError, AttributeError):
                msg = f"HTTP {status}"
            return {"enabled": True, "ok": False, "error": f"GitHub API error: {msg}"}

        release = json.loads(body)
        latest_tag = release.get("tag_name", "")
        latest_version = latest_tag.lstrip("vV")
        is_newer = _parse_version(latest_tag) > _parse_version(self.current_version)

        dmg_asset = next(
            (a for a in release.get("assets", []) if a["name"].endswith(".dmg")), None
        )
        return {
            "enabled": True,
            "ok": True,
            "current_version": self.current_version,
            "latest_version": latest_version,
            "update_available": is_newer,
            "notes": release.get("body", ""),
            "published_at": release.get("published_at"),
            "asset_name": dmg_asset["name"] if dmg_asset else None,
            "asset_api_url": dmg_asset["url"] if dmg_asset else None,
            "asset_size": dmg_asset["size"] if dmg_asset else None,
        }

    def download(self, asset_api_url: str, dest_path: str,
                 progress_cb=None) -> str:
        """Download a private release asset by its API `url` (not
        `browser_download_url`, which 404s without auth on a private repo).
        Requires the octet-stream Accept header to get raw bytes instead of
        the asset's JSON metadata.
        """
        if not self._token:
            raise RuntimeError("update checks not configured for this build")
        req = urllib.request.Request(asset_api_url, headers={
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/octet-stream",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb:
                        progress_cb(downloaded, total)
        return dest_path
