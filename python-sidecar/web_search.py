"""Web search — the one capability in Aria that sends data off this Mac.

Everything else in this app is explicitly offline-first (memory, skills,
persona, chat). Search is different by necessity, so it's held to a higher
bar: off by default (registered as a Tool with ``enabled=False``, same
pattern as ``run_shell``), and invoked deterministically — see
``extract_search_query`` — rather than trusting the model to correctly emit
a native function-call. That native path is unreliable here anyway: the
VLM-mode driver used by the recommended 12B/e4b checkpoints never even
receives tool specs (see engine/mlx_driver.py's ``_generate_vlm``).

Uses DuckDuckGo's HTML endpoint because it needs no API key or signup —
consistent with every other capability in Aria shipping zero-config.
Parsing is regex-based against DDG's stable-ish result markup; if that
markup changes this degrades to an empty result list, not a crash.
"""
from __future__ import annotations

import html
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request


def _ssl_context() -> ssl.SSLContext:
    """Explicit certifi-backed context — same fix as update_checker.py's
    _ssl_context(). Confirmed live: without this, urlopen fails with
    CERTIFICATE_VERIFY_FAILED against DuckDuckGo on a framework Python build
    whose default trust store isn't populated, silently degrading every
    search to "no results" instead of actually searching."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()

_RESULT_RE = re.compile(
    r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>.*?'
    r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
    re.DOTALL,
)

# Deliberately conservative, high-precision phrasing — same philosophy as
# memory.extract_auto_facts — so this never fires on ordinary conversation.
_SEARCH_TRIGGER_RE = re.compile(
    r"^\s*(?:please\s+)?(?:search(?:\s+the\s+web)?(?:\s+for)?|look\s+up|google)\s*[:\-]?\s*(.{2,300})",
    re.I,
)


def _strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def extract_search_query(text: str) -> str | None:
    """Pull a search query out of an explicit "search for .../look up .../
    google ..." request. Returns None for anything else — this never
    guesses at intent from ordinary phrasing."""
    m = _SEARCH_TRIGGER_RE.match(text or "")
    if not m:
        return None
    q = m.group(1).strip().rstrip("?.")
    return q or None


def search_web(query: str, k: int = 5, timeout: int = 8) -> list[dict]:
    """Best-effort web search via DuckDuckGo's HTML endpoint.

    Returns [] on any network or parse failure rather than raising — a
    flaky connection or a DDG markup change should degrade the chat turn
    to "no results found", not break it.
    """
    query = (query or "").strip()
    if not query:
        return []
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
    # A bare "User-Agent: Mozilla/5.0" (confirmed live) gets served DDG's
    # bot-detection challenge page instead of results — no result markup at
    # all, just an anomaly.js challenge form. A header set that actually
    # resembles a real desktop browser avoids it.
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://html.duckduckgo.com/",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, ValueError):
        return []

    results = []
    for m in _RESULT_RE.finditer(body):
        href, title, snippet = m.groups()
        results.append({
            "title": _strip_tags(title),
            "url": html.unescape(href),
            "snippet": _strip_tags(snippet),
        })
        if len(results) >= k:
            break
    return results
