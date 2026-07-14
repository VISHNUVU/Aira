"""Chat history — persisted conversations, listed in the sidebar.

Every chat turn (through chat()/chat_stream() in app.py) is saved here so
closing and reopening Aria never loses a conversation. A session's title is
set once, from its first user message, the moment that message is saved —
matches how ChatGPT/Claude name new chats, and means the sidebar never shows
a stale "New chat" placeholder once there's real content to name it from.
"""
from __future__ import annotations

import json
import uuid
from typing import Optional

from store import Store, now_ms

TITLE_MAX_CHARS = 60


def _title_from_text(text: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "New chat"
    return text[:TITLE_MAX_CHARS] + ("…" if len(text) > TITLE_MAX_CHARS else "")


class ChatHistory:
    def __init__(self, store: Store):
        self.store = store

    def create_session(self, title: Optional[str] = None) -> dict:
        sid = str(uuid.uuid4())
        now = now_ms()
        self.store.execute(
            "INSERT INTO chat_sessions (id,title,created_at,updated_at) VALUES (?,?,?,?)",
            (sid, (title or "New chat").strip() or "New chat", now, now),
        )
        return self.get_session(sid)

    def get_session(self, session_id: str) -> Optional[dict]:
        rows = self.store.query("SELECT * FROM chat_sessions WHERE id=?", (session_id,))
        return dict(rows[0]) if rows else None

    def list_sessions(self, limit: int = 200) -> list[dict]:
        rows = self.store.query(
            """SELECT s.*, COUNT(m.id) AS message_count
               FROM chat_sessions s LEFT JOIN chat_messages m ON m.session_id = s.id
               GROUP BY s.id ORDER BY s.updated_at DESC LIMIT ?""",
            (limit,),
        )
        return [dict(r) for r in rows]

    def get_messages(self, session_id: str) -> list[dict]:
        rows = [dict(r) for r in self.store.query(
            "SELECT * FROM chat_messages WHERE session_id=? ORDER BY created_at ASC",
            (session_id,))]
        for r in rows:
            r["image_job"] = json.loads(r["image_job"]) if r.get("image_job") else None
        return rows

    def add_message(self, session_id: str, role: str, content: str,
                     image_job: Optional[dict] = None) -> dict:
        content = (content or "").strip()
        if not content:
            return {}
        mid = str(uuid.uuid4())
        now = now_ms()
        self.store.execute(
            "INSERT INTO chat_messages (id,session_id,role,content,created_at,image_job) "
            "VALUES (?,?,?,?,?,?)",
            (mid, session_id, role, content, now,
             json.dumps(image_job) if image_job is not None else None),
        )
        # First message in the session names it — every later message just
        # bumps updated_at so the sidebar sorts by most-recently-active.
        existing = self.get_messages(session_id)
        if role == "user" and len(existing) == 1:
            self.store.execute(
                "UPDATE chat_sessions SET title=?, updated_at=? WHERE id=?",
                (_title_from_text(content), now, session_id))
        else:
            self.store.execute(
                "UPDATE chat_sessions SET updated_at=? WHERE id=?", (now, session_id))
        return {"id": mid, "session_id": session_id, "role": role,
                "content": content, "created_at": now, "image_job": image_job}

    def update_message_image_job(self, message_id: str, image_job: dict) -> None:
        """Called once a background image-generation job finishes (see
        SidecarService.image_generate) — replaces the "still running" job
        snapshot saved at message-creation time with the final result, so
        reopening this chat later shows the actual image instead of a
        placeholder stuck polling a long-finished job."""
        self.store.execute(
            "UPDATE chat_messages SET image_job=? WHERE id=?",
            (json.dumps(image_job), message_id))

    def rename_session(self, session_id: str, title: str) -> dict:
        title = (title or "").strip()
        if not title:
            raise ValueError("title cannot be empty")
        self.store.execute(
            "UPDATE chat_sessions SET title=? WHERE id=?", (title, session_id))
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> None:
        self.store.execute("DELETE FROM chat_messages WHERE session_id=?", (session_id,))
        self.store.execute("DELETE FROM chat_sessions WHERE id=?", (session_id,))

    def ensure_session(self, session_id: Optional[str], first_user_message: str = "") -> str:
        """Returns a valid session id — the one passed in if it still exists,
        otherwise a freshly created one. Used at the top of every chat turn
        so the caller never has to special-case "no session yet"."""
        if session_id and self.get_session(session_id):
            return session_id
        return self.create_session(_title_from_text(first_user_message))["id"]
