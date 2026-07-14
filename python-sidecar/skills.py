"""Skills — user-authored instruction templates, created by chatting with
Aria or typed directly in Settings.

A skill is deliberately NOT executable code — that's a much bigger security
surface (this project already ships `run_shell` disabled by default for
exactly that reason). A skill is just a named block of instructions Aria
follows for a turn whenever its trigger phrase shows up in the user's
message — the same additive-context mechanism as the persona prompt and RAG
memory context in app.py, not a new capability the model can act on its own.
"""
from __future__ import annotations

import re
import uuid
from typing import Optional

from store import Store, now_ms

# Parses the structured draft the model is asked to produce in
# SidecarService.skills_draft(). INSTRUCTIONS is last and greedy so it can
# span multiple lines; NAME/TRIGGER are single-line.
_DRAFT_RE = re.compile(
    r"NAME:\s*(.+?)\s*\n\s*TRIGGER:\s*(.+?)\s*\n\s*INSTRUCTIONS:\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)


def parse_skill_draft(text: str) -> dict:
    m = _DRAFT_RE.search(text)
    if not m:
        return {"ok": False, "error": "Couldn't parse a skill from that reply — try rephrasing the description.", "raw": text}
    name, trigger, instructions = (g.strip() for g in m.groups())
    if not name or not instructions:
        return {"ok": False, "error": "Model left name or instructions empty — try rephrasing.", "raw": text}
    return {"ok": True, "name": name, "trigger": trigger, "instructions": instructions}


class SkillLibrary:
    """Stores skills and matches them against incoming chat messages."""

    def __init__(self, store: Store):
        self.store = store

    def add(self, name: str, instructions: str, trigger: Optional[str] = None) -> dict:
        name = (name or "").strip()
        instructions = (instructions or "").strip()
        if not name or not instructions:
            raise ValueError("name and instructions are required")
        sid = str(uuid.uuid4())
        self.store.execute(
            "INSERT INTO skills (id,name,trigger,instructions,created_at) VALUES (?,?,?,?,?)",
            (sid, name, (trigger or "").strip() or None, instructions, now_ms()),
        )
        return self.get(sid)

    def list(self) -> list[dict]:
        return [dict(r) for r in self.store.query(
            "SELECT * FROM skills ORDER BY created_at DESC")]

    def get(self, skill_id: str) -> Optional[dict]:
        rows = self.store.query("SELECT * FROM skills WHERE id=?", (skill_id,))
        return dict(rows[0]) if rows else None

    def delete(self, skill_id: str) -> None:
        self.store.execute("DELETE FROM skills WHERE id=?", (skill_id,))

    def match(self, text: str) -> list[dict]:
        """Skills whose trigger phrase appears in the user's message —
        deterministic substring matching, not LLM-guessed intent, so it's
        predictable which skill (if any) fires for a given message."""
        text_l = (text or "").lower()
        return [s for s in self.list() if s["trigger"] and s["trigger"].lower() in text_l]
