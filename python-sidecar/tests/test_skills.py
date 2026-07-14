"""Tests for the skills module — user-authored instruction templates."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from store import Store
from skills import SkillLibrary, parse_skill_draft


def make_lib():
    return SkillLibrary(Store(":memory:"))


def test_add_and_get():
    lib = make_lib()
    s = lib.add("Meeting Notes", "Turn rough notes into a structured summary.",
                trigger="meeting notes")
    assert s["name"] == "Meeting Notes"
    assert s["trigger"] == "meeting notes"
    got = lib.get(s["id"])
    assert got["instructions"] == "Turn rough notes into a structured summary."


def test_add_requires_name_and_instructions():
    lib = make_lib()
    try:
        lib.add("", "instructions")
        assert False, "should have raised"
    except ValueError:
        pass
    try:
        lib.add("Name", "")
        assert False, "should have raised"
    except ValueError:
        pass


def test_add_without_trigger():
    lib = make_lib()
    s = lib.add("Untriggered", "Some instructions")
    assert s["trigger"] is None


def test_list_ordered_newest_first():
    lib = make_lib()
    a = lib.add("First", "a")
    b = lib.add("Second", "b")
    listed = lib.list()
    assert [s["id"] for s in listed] == [b["id"], a["id"]]


def test_delete():
    lib = make_lib()
    s = lib.add("Temp", "instructions")
    lib.delete(s["id"])
    assert lib.get(s["id"]) is None


def test_match_case_insensitive_substring():
    lib = make_lib()
    lib.add("Meeting Notes", "summarize notes", trigger="meeting notes")
    lib.add("Email Draft", "draft an email", trigger="draft email")
    hits = lib.match("Can you do MEETING NOTES for this?")
    assert len(hits) == 1 and hits[0]["name"] == "Meeting Notes"


def test_match_no_trigger_never_matches():
    lib = make_lib()
    lib.add("Silent", "instructions")  # no trigger
    assert lib.match("silent please") == []


def test_match_multiple_skills():
    lib = make_lib()
    lib.add("A", "a", trigger="alpha")
    lib.add("B", "b", trigger="beta")
    hits = lib.match("please run alpha and beta")
    assert {h["name"] for h in hits} == {"A", "B"}


def test_parse_skill_draft_well_formed():
    text = (
        "NAME: Meeting Notes\n"
        "TRIGGER: meeting notes\n"
        "INSTRUCTIONS: Turn rough notes into a structured summary with "
        "action items, across multiple lines\nif needed."
    )
    r = parse_skill_draft(text)
    assert r["ok"] is True
    assert r["name"] == "Meeting Notes"
    assert r["trigger"] == "meeting notes"
    assert "action items" in r["instructions"]


def test_parse_skill_draft_malformed():
    r = parse_skill_draft("I don't understand the request.")
    assert r["ok"] is False
    assert "raw" in r


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
