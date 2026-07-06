"""Update checking against a self-hosted manifest on the project's own VPS.

Design: "check + download, manual install" — not a silent auto-updater.
Replaces an earlier GitHub-Releases-based version: that repo is private, so
checking it needed a scoped PAT baked into the shipped app plus a two-step
authenticated fetch (GitHub's asset API requires resolving an asset's `url`
before it can be downloaded). Self-hosting the manifest removes the need for
any embedded credential at all — the endpoint is just a plain public URL.

Known limitation: the VPS currently serves this over plain HTTP (no domain
yet for a Let's Encrypt cert), so there's no transport encryption and no
code-signing check on the downloaded binary. SHA-256 verification (below)
catches corruption and accidental mismatches, but NOT a deliberate
on-path attacker substituting a malicious build — that needs HTTPS (and
ideally Tauri's ed25519 update-signature verification) to actually close.
Fine for now as a "check + manual install" flow where you still look at
what you're opening; worth hardening before this become a silent installer.
"""
from __future__ import annotations

import hashlib
import json
import os
import ssl
import urllib.error
import urllib.request

MANIFEST_URL = "http://76.13.244.25:8099/releases/latest.json"


def _ssl_context() -> ssl.SSLContext:
    """Explicit certifi-backed context instead of relying on the system's
    default cert store — python.org/framework macOS builds commonly ship
    without their OpenSSL cert path populated (the "Install Certificates
    .command" step), which makes urlopen fail with CERTIFICATE_VERIFY_FAILED
    even though the machine's own trust store is fine."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _parse_version(v: str) -> tuple:
    """'v1.2.3' or '1.2.3' -> (1, 2, 3), for a simple numeric comparison."""
    cleaned = v.lstrip("vV")
    parts = []
    for p in cleaned.split("."):
        digits = "".join(c for c in p if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


class UpdateChecker:
    def __init__(self, current_version: str, manifest_url: str = MANIFEST_URL):
        self.current_version = current_version
        self.manifest_url = manifest_url

    @property
    def available(self) -> bool:
        return True  # no credential/config needed — always attemptable

    def check(self) -> dict:
        """Fetch the manifest and compare its version to current_version."""
        try:
            req = urllib.request.Request(self.manifest_url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10, context=_ssl_context()) as resp:
                manifest = json.loads(resp.read())
        except urllib.error.URLError as e:
            # Includes "offline" — this app is meant to work with no network
            # connection, so a failed update check must never be fatal.
            return {"enabled": True, "ok": False, "error": f"couldn't reach update server: {e.reason}"}
        except (json.JSONDecodeError, ValueError) as e:
            return {"enabled": True, "ok": False, "error": f"bad manifest: {e}"}

        latest_version = str(manifest.get("version", ""))
        is_newer = _parse_version(latest_version) > _parse_version(self.current_version)
        return {
            "enabled": True,
            "ok": True,
            "current_version": self.current_version,
            "latest_version": latest_version,
            "update_available": is_newer,
            "notes": manifest.get("notes", ""),
            "published_at": manifest.get("published_at"),
            "asset_name": manifest.get("asset_name"),
            "asset_url": manifest.get("asset_url"),
            "asset_size": manifest.get("size"),
            "asset_sha256": manifest.get("sha256"),
        }

    def download(self, asset_url: str, dest_path: str,
                 expected_sha256: str | None = None, progress_cb=None) -> str:
        """Download an update asset, verifying its checksum if provided."""
        req = urllib.request.Request(asset_url)
        hasher = hashlib.sha256()
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
                    hasher.update(chunk)
                    downloaded += len(chunk)
                    if progress_cb:
                        progress_cb(downloaded, total)

        if expected_sha256 and hasher.hexdigest() != expected_sha256:
            os.remove(dest_path)
            raise ValueError(
                f"downloaded file failed checksum verification "
                f"(expected {expected_sha256[:12]}…, got {hasher.hexdigest()[:12]}…) — "
                f"deleted, not installing a possibly-corrupted/tampered build"
            )
        return dest_path
