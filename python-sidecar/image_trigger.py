"""Deterministic trigger detection for "generate me an image" requests —
same pattern as web_search.py's extract_search_query: regex-based phrase
matching done server-side, not native model tool-calling. Native
function-calling is confirmed unreliable elsewhere in this app (Gemma
leaks raw tool-call syntax into visible replies for certain phrasings —
see engine/mlx_driver.py's _filter_tool_call_leak) — this sidesteps that
entirely by never asking the model to decide, just scanning the user's own
words with a phrase pattern.
"""
from __future__ import annotations

import re

_IMAGE_TRIGGER_RE = re.compile(
    r"^\s*(?:please\s+)?(?:generate|create|draw|make|paint|render)\s+(?:me\s+)?"
    # "an?\s*" then an optional stray "a" tolerates a common typo where the
    # article gets glued onto the noun ("an aimage", "a apicture") instead
    # of leaving a space — without this, the whole deterministic trigger
    # misses and the request falls through to the plain chat model, which
    # then hallucinates a text refusal instead of actually generating.
    r"(?:an?\s*)?a?(?:image|picture|photo|drawing|illustration|painting)\s+(?:of\s+)?(.{2,400})",
    re.I,
)


def extract_image_prompt(text: str) -> str | None:
    m = _IMAGE_TRIGGER_RE.match(text or "")
    if not m:
        return None
    prompt = m.group(1).strip().rstrip("?.")
    return prompt or None
