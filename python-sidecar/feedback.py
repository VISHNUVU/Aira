"""Feedback store + training-example pipeline.

Captures user feedback (thumbs-up / edit / correction), normalizes it into
instruction->preferred-response pairs, deduplicates, and persists to the
``examples`` table. Provides the trigger logic that tells the trainer when
enough new validated examples have accumulated, and exports them to JSONL in the
Gemma 4 chat format.

Fine-tuning teaches behavior/style/format — so we only keep *validated*
examples (an explicit thumbs-up, an edited answer, or a correction), never raw
unlabelled turns.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Optional

from store import Store, now_ms

# Feedback types that produce a training example.
FEEDBACK_TYPES = {"thumbs_up", "edit", "correction"}


def _dedup_hash(instruction: str, preferred: str) -> str:
    norm = (instruction.strip() + "\x00" + preferred.strip()).lower()
    return hashlib.sha256(norm.encode()).hexdigest()


@dataclass
class Example:
    id: str
    instruction: str
    preferred_output: str
    feedback_type: str
    context: Optional[str] = None
    rejected_output: Optional[str] = None
    source_turn_id: Optional[str] = None


class FeedbackStore:
    """Owns the ``examples`` table and the train-trigger logic."""

    def __init__(self, store: Store, train_threshold: int = 200):
        self.store = store
        self.train_threshold = train_threshold

    # ---- capture ---------------------------------------------------------
    def capture(
        self,
        instruction: str,
        preferred_output: str,
        feedback_type: str,
        *,
        context: Optional[str] = None,
        rejected_output: Optional[str] = None,
        source_turn_id: Optional[str] = None,
    ) -> Optional[str]:
        """Record one feedback event as a training example.

        Returns the new example id, or ``None`` if it was a duplicate.
        Raises ValueError on invalid input.
        """
        if feedback_type not in FEEDBACK_TYPES:
            raise ValueError(
                f"unknown feedback_type {feedback_type!r}; "
                f"expected one of {sorted(FEEDBACK_TYPES)}"
            )
        instruction = (instruction or "").strip()
        preferred_output = (preferred_output or "").strip()
        if not instruction or not preferred_output:
            raise ValueError("instruction and preferred_output must be non-empty")

        dh = _dedup_hash(instruction, preferred_output)
        eid = str(uuid.uuid4())
        try:
            self.store.conn.execute(
                "INSERT INTO examples (id,instruction,context,preferred_output,"
                "rejected_output,feedback_type,dedup_hash,used_in_run,split,"
                "created_at,source_turn_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (eid, instruction, context, preferred_output, rejected_output,
                 feedback_type, dh, None, None, now_ms(), source_turn_id),
            )
            self.store.conn.commit()
            return eid
        except Exception as e:  # sqlite3.IntegrityError on dedup_hash UNIQUE
            if "UNIQUE" in str(e):
                return None
            raise

    # ---- trigger ---------------------------------------------------------
    def unused_count(self) -> int:
        return self.store.query(
            "SELECT COUNT(*) AS n FROM examples WHERE used_in_run IS NULL"
        )[0]["n"]

    def should_train(self) -> bool:
        return self.unused_count() >= self.train_threshold

    # ---- export ----------------------------------------------------------
    def export_jsonl(self, path: str, only_unused: bool = True) -> int:
        """Write examples to JSONL in Gemma 4 chat format. Returns row count."""
        where = "WHERE used_in_run IS NULL" if only_unused else ""
        rows = self.store.query(
            f"SELECT id,instruction,context,preferred_output FROM examples "
            f"{where} ORDER BY created_at ASC"
        )
        n = 0
        with open(path, "w") as f:
            for r in rows:
                user = r["instruction"]
                if r["context"]:
                    user = f"{r['context']}\n\n{user}"
                obj = {"messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": r["preferred_output"]},
                ]}
                f.write(json.dumps(obj) + "\n")
                n += 1
        return n

    def mark_used(self, run_id: str, example_ids: Optional[list[str]] = None) -> int:
        """Mark examples as consumed by a training run."""
        if example_ids is None:
            cur = self.store.conn.execute(
                "UPDATE examples SET used_in_run=? WHERE used_in_run IS NULL",
                (run_id,),
            )
        else:
            ph = ",".join("?" for _ in example_ids)
            cur = self.store.conn.execute(
                f"UPDATE examples SET used_in_run=? WHERE id IN ({ph})",
                (run_id, *example_ids),
            )
        self.store.conn.commit()
        return cur.rowcount

    # ---- introspection ---------------------------------------------------
    def list_examples(self, limit: int = 200, pending_only: bool = False) -> list[dict]:
        where = "WHERE used_in_run IS NULL" if pending_only else ""
        rows = self.store.query(
            f"SELECT id,substr(instruction,1,120) AS instruction,"
            f"substr(preferred_output,1,160) AS preferred_output,feedback_type,"
            f"used_in_run,created_at FROM examples {where} "
            f"ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        total = self.store.query("SELECT COUNT(*) AS n FROM examples")[0]["n"]
        pending = self.unused_count()
        by_type = {
            r["feedback_type"]: r["n"] for r in self.store.query(
                "SELECT feedback_type, COUNT(*) AS n FROM examples "
                "GROUP BY feedback_type")
        }
        return {
            "total": total, "pending": pending,
            "threshold": self.train_threshold,
            "ready_to_train": pending >= self.train_threshold,
            "by_type": by_type,
        }
