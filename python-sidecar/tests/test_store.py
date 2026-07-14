"""Tests for store.py — schema creation and in-place migrations.

The image_job migration matters because CREATE TABLE IF NOT EXISTS is a
no-op on a database that already has chat_messages from before that column
existed (i.e. every real ~/.aria/aria.db predating this feature) — without
an explicit ALTER TABLE, those installs would silently keep failing to
persist generated images forever.
"""
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from store import Store


def test_fresh_db_has_image_job_column():
    s = Store(":memory:")
    cols = [r["name"] for r in s.conn.execute("PRAGMA table_info(chat_messages)")]
    assert "image_job" in cols


def test_existing_db_without_image_job_gets_migrated():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "aria.db")
        # Simulate a real pre-existing database from before image_job existed.
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE chat_messages (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT NOT NULL, created_at INTEGER NOT NULL)""")
        conn.execute("INSERT INTO chat_messages VALUES ('m1','s1','user','hi',1)")
        conn.commit()
        conn.close()

        s = Store(db_path)
        cols = [r["name"] for r in s.conn.execute("PRAGMA table_info(chat_messages)")]
        assert "image_job" in cols
        # Pre-existing row survives the migration, new column reads as NULL.
        row = s.conn.execute("SELECT * FROM chat_messages WHERE id='m1'").fetchone()
        assert row["content"] == "hi"
        assert row["image_job"] is None


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} tests passed")
    sys.exit(0 if passed == len(fns) else 1)
