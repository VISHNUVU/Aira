"""Tests for chat_history.py — session CRUD and auto-titling."""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from store import Store
from chat_history import ChatHistory


def fresh_history():
    return ChatHistory(Store(":memory:"))


def test_create_session_defaults_to_new_chat_title():
    h = fresh_history()
    s = h.create_session()
    assert s["title"] == "New chat"
    assert s["id"]


def test_first_user_message_sets_session_title():
    h = fresh_history()
    s = h.create_session()
    h.add_message(s["id"], "user", "What's the capital of France?")
    updated = h.get_session(s["id"])
    assert updated["title"] == "What's the capital of France?"


def test_long_first_message_title_is_truncated():
    h = fresh_history()
    s = h.create_session()
    long_text = "x" * 200
    h.add_message(s["id"], "user", long_text)
    updated = h.get_session(s["id"])
    assert len(updated["title"]) <= 61  # 60 chars + ellipsis
    assert updated["title"].endswith("…")


def test_second_message_does_not_change_title():
    h = fresh_history()
    s = h.create_session()
    h.add_message(s["id"], "user", "first message")
    h.add_message(s["id"], "assistant", "a reply")
    h.add_message(s["id"], "user", "second message")
    updated = h.get_session(s["id"])
    assert updated["title"] == "first message"


def test_get_messages_returns_in_order():
    h = fresh_history()
    s = h.create_session()
    h.add_message(s["id"], "user", "one")
    h.add_message(s["id"], "assistant", "two")
    h.add_message(s["id"], "user", "three")
    msgs = h.get_messages(s["id"])
    assert [m["content"] for m in msgs] == ["one", "two", "three"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]


def test_list_sessions_sorted_by_most_recently_active():
    h = fresh_history()
    a = h.create_session("A")
    time.sleep(0.005)
    b = h.create_session("B")
    time.sleep(0.005)
    h.add_message(a["id"], "user", "touch A")  # bumps A's updated_at last
    sessions = h.list_sessions()
    assert sessions[0]["id"] == a["id"]
    assert sessions[1]["id"] == b["id"]


def test_list_sessions_includes_message_count():
    h = fresh_history()
    s = h.create_session()
    h.add_message(s["id"], "user", "hi")
    h.add_message(s["id"], "assistant", "hello")
    sessions = h.list_sessions()
    assert sessions[0]["message_count"] == 2


def test_rename_session():
    h = fresh_history()
    s = h.create_session()
    renamed = h.rename_session(s["id"], "My renamed chat")
    assert renamed["title"] == "My renamed chat"


def test_rename_session_rejects_empty_title():
    h = fresh_history()
    s = h.create_session()
    try:
        h.rename_session(s["id"], "   ")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_delete_session_removes_messages_too():
    h = fresh_history()
    s = h.create_session()
    h.add_message(s["id"], "user", "hi")
    h.delete_session(s["id"])
    assert h.get_session(s["id"]) is None
    assert h.get_messages(s["id"]) == []


def test_ensure_session_creates_when_none_given():
    h = fresh_history()
    sid = h.ensure_session(None, "hello there")
    assert h.get_session(sid) is not None


def test_ensure_session_reuses_existing_valid_id():
    h = fresh_history()
    s = h.create_session()
    sid = h.ensure_session(s["id"], "irrelevant")
    assert sid == s["id"]


def test_ensure_session_creates_fresh_when_id_unknown():
    h = fresh_history()
    sid = h.ensure_session("does-not-exist", "hello")
    assert sid != "does-not-exist"
    assert h.get_session(sid) is not None


def test_add_message_ignores_blank_content():
    h = fresh_history()
    s = h.create_session()
    result = h.add_message(s["id"], "user", "   ")
    assert result == {}
    assert h.get_messages(s["id"]) == []


def test_message_without_image_job_round_trips_as_none():
    h = fresh_history()
    s = h.create_session()
    h.add_message(s["id"], "user", "hi")
    msgs = h.get_messages(s["id"])
    assert msgs[0]["image_job"] is None


def test_add_message_persists_image_job():
    h = fresh_history()
    s = h.create_session()
    job = {"ok": True, "status": "running", "prompt": "a cat"}
    saved = h.add_message(s["id"], "assistant", "Generating...", image_job=job)
    assert saved["image_job"] == job
    msgs = h.get_messages(s["id"])
    assert msgs[0]["image_job"] == job


def test_update_message_image_job_replaces_stored_job():
    h = fresh_history()
    s = h.create_session()
    saved = h.add_message(s["id"], "assistant", "Generating...",
                           image_job={"status": "running"})
    done_job = {"status": "done", "result": {"url": "/images/file?filename=x.png"}}
    h.update_message_image_job(saved["id"], done_job)
    msgs = h.get_messages(s["id"])
    assert msgs[0]["image_job"] == done_job


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
