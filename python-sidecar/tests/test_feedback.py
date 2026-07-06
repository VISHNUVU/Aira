"""Tests for the feedback store + training-example pipeline."""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from store import Store
from feedback import FeedbackStore, _dedup_hash


def make_fb(threshold=5):
    return FeedbackStore(Store(":memory:"), train_threshold=threshold)


def test_capture_basic():
    fb = make_fb()
    eid = fb.capture("What is 2+2?", "4", "thumbs_up")
    assert eid is not None
    assert fb.unused_count() == 1


def test_capture_validates_type():
    fb = make_fb()
    try:
        fb.capture("q", "a", "bogus_type")
        assert False, "should have raised"
    except ValueError:
        pass


def test_capture_rejects_empty():
    fb = make_fb()
    for bad in [("", "a"), ("q", ""), ("  ", "a")]:
        try:
            fb.capture(*bad, "edit")
            assert False, f"should reject {bad}"
        except ValueError:
            pass


def test_dedup():
    fb = make_fb()
    e1 = fb.capture("same question", "same answer", "thumbs_up")
    e2 = fb.capture("same question", "same answer", "correction")  # dup content
    assert e1 is not None
    assert e2 is None, "duplicate should return None"
    assert fb.unused_count() == 1
    # different content is not a dup
    e3 = fb.capture("same question", "different answer", "edit")
    assert e3 is not None
    assert fb.unused_count() == 2


def test_dedup_hash_normalizes():
    assert _dedup_hash("Hello ", " World") == _dedup_hash("hello", "world")


def test_trigger_threshold():
    fb = make_fb(threshold=3)
    assert not fb.should_train()
    for i in range(2):
        fb.capture(f"q{i}", f"a{i}", "thumbs_up")
    assert not fb.should_train()  # 2 < 3
    fb.capture("q2", "a2", "thumbs_up")
    assert fb.should_train()      # 3 >= 3


def test_export_jsonl_format():
    fb = make_fb()
    fb.capture("Summarize X", "X is a thing.", "edit", context="Doc about X")
    fb.capture("What color?", "Blue.", "thumbs_up")
    path = "/tmp/aria_export_test.jsonl"
    n = fb.export_jsonl(path)
    assert n == 2
    with open(path) as f:
        rows = [json.loads(ln) for ln in f]
    assert rows[0]["messages"][0]["role"] == "user"
    assert rows[0]["messages"][1]["role"] == "assistant"
    # context is prepended to the user turn
    assert "Doc about X" in rows[0]["messages"][0]["content"]
    assert rows[0]["messages"][1]["content"] == "X is a thing."


def test_mark_used_excludes_from_pending():
    fb = make_fb(threshold=100)
    for i in range(4):
        fb.capture(f"q{i}", f"a{i}", "thumbs_up")
    assert fb.unused_count() == 4
    marked = fb.mark_used("run-123")
    assert marked == 4
    assert fb.unused_count() == 0
    # a new example after marking is pending again
    fb.capture("new q", "new a", "edit")
    assert fb.unused_count() == 1


def test_stats():
    fb = make_fb(threshold=2)
    fb.capture("a", "1", "thumbs_up")
    fb.capture("b", "2", "edit")
    fb.capture("c", "3", "thumbs_up")
    s = fb.stats()
    assert s["total"] == 3
    assert s["pending"] == 3
    assert s["ready_to_train"] is True
    assert s["by_type"]["thumbs_up"] == 2
    assert s["by_type"]["edit"] == 1


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
