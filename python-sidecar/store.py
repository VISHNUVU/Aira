"""SQLite store — the single metadata database for Aria.

Owns schema creation + migrations for every table defined in docs/schemas.md:
memory_chunks, examples, adapters, training_runs, tool_calls, meta.

All other modules (memory, feedback, trainer, tools) take a Store instance and
use its connection. Pure stdlib (sqlite3) — no dependencies, fully testable.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Optional

SCHEMA_VERSION = 1


def now_ms() -> int:
    return int(time.time() * 1000)


SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_chunks (
    id              TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    text            TEXT NOT NULL,
    token_count     INTEGER,
    chunk_index     INTEGER,
    embedding_model TEXT NOT NULL,
    created_at      INTEGER NOT NULL,
    retrieval_count INTEGER NOT NULL DEFAULT 0,
    last_retrieved  INTEGER,
    metadata        TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_source  ON memory_chunks(source);
CREATE INDEX IF NOT EXISTS idx_memory_created ON memory_chunks(created_at);

CREATE TABLE IF NOT EXISTS examples (
    id               TEXT PRIMARY KEY,
    instruction      TEXT NOT NULL,
    context          TEXT,
    preferred_output TEXT NOT NULL,
    rejected_output  TEXT,
    feedback_type    TEXT NOT NULL,
    dedup_hash       TEXT NOT NULL,
    used_in_run      TEXT,
    split            TEXT,
    created_at       INTEGER NOT NULL,
    source_turn_id   TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_examples_dedup ON examples(dedup_hash);

CREATE TABLE IF NOT EXISTS adapters (
    id             TEXT PRIMARY KEY,
    base_model     TEXT NOT NULL,
    path           TEXT NOT NULL,
    training_run   TEXT NOT NULL,
    eval_score     REAL,
    baseline_score REAL,
    promoted       INTEGER NOT NULL,
    is_active      INTEGER NOT NULL DEFAULT 0,
    n_examples     INTEGER,
    created_at     INTEGER NOT NULL,
    notes          TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_adapters_active
    ON adapters(is_active) WHERE is_active = 1;

CREATE TABLE IF NOT EXISTS training_runs (
    id             TEXT PRIMARY KEY,
    status         TEXT NOT NULL,
    base_model     TEXT NOT NULL,
    n_train        INTEGER,
    n_held_out     INTEGER,
    config         TEXT,
    train_loss     TEXT,
    eval_score     REAL,
    baseline_score REAL,
    adapter_id     TEXT,
    started_at     INTEGER,
    finished_at    INTEGER,
    error          TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON training_runs(status);

CREATE TABLE IF NOT EXISTS tool_calls (
    id           TEXT PRIMARY KEY,
    tool_name    TEXT NOT NULL,
    arguments    TEXT NOT NULL,
    result       TEXT,
    status       TEXT NOT NULL,
    duration_ms  INTEGER,
    turn_id      TEXT,
    created_at   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_toolcalls_created ON tool_calls(created_at);
CREATE INDEX IF NOT EXISTS idx_toolcalls_name    ON tool_calls(tool_name);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Store:
    """Thin wrapper over a sqlite3 connection with schema management."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self._migrate()

    def _migrate(self) -> None:
        self.conn.executescript(SCHEMA)
        cur = self.get_meta("schema_version")
        if cur is None:
            self.set_meta("schema_version", str(SCHEMA_VERSION))
        self.conn.commit()

    # ---- meta helpers ----------------------------------------------------
    def get_meta(self, key: str, default: Optional[str] = None) -> Optional[str]:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    def get_json(self, key: str, default: Any = None) -> Any:
        v = self.get_meta(key)
        return json.loads(v) if v is not None else default

    def set_json(self, key: str, value: Any) -> None:
        self.set_meta(key, json.dumps(value))

    # ---- generic helpers -------------------------------------------------
    def execute(self, sql: str, params: tuple = ()):  # convenience passthrough
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def close(self) -> None:
        self.conn.close()
