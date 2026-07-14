"""Tests for mlx_driver's streaming output filters.

These filters are plain string/regex generators with no mlx_lm dependency,
so they're importable and testable even off Apple Silicon (the driver
CLASS's methods lazily import mlx_lm only when actually called).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.mlx_driver import _filter_channel_tags, _filter_tool_call_leak


def run_filter(fn, chunks):
    return "".join(fn(iter(chunks)))


def test_channel_tag_stripped_single_chunk():
    out = run_filter(_filter_channel_tags, ["<|channel>thought<channel|>Hello there"])
    assert out == "Hello there"


def test_channel_tag_stripped_across_chunk_boundary():
    out = run_filter(_filter_channel_tags, ["<|channel>tho", "ught<channel|>Hi"])
    assert out == "Hi"


def test_channel_tag_absent_passes_through():
    out = run_filter(_filter_channel_tags, ["Just a normal reply."])
    assert out == "Just a normal reply."


def test_tool_call_leak_truncates_with_no_preamble():
    leaked = '<|tool_call>call:web_search{queries:[<|"|>x<|"|>]}<tool_call|>'
    out = run_filter(_filter_tool_call_leak, [leaked])
    assert out == ""


def test_tool_call_leak_keeps_preamble_before_it():
    text = ('I need to search for that. Please bear with me.'
            '<|tool_call>call:web_search{queries:[<|"|>x<|"|>]}<tool_call|>')
    out = run_filter(_filter_tool_call_leak, [text])
    assert out == "I need to search for that. Please bear with me."


def test_tool_call_leak_split_across_chunks():
    # the literal tag boundary itself split across two streamed chunks
    out = run_filter(_filter_tool_call_leak, ["some reply<|tool_", 'call>rest of garbage'])
    assert out == "some reply"


def test_tool_call_leak_absent_passes_through_unchanged():
    out = run_filter(_filter_tool_call_leak, ["A perfectly ordinary reply, nothing to see here."])
    assert out == "A perfectly ordinary reply, nothing to see here."


def test_tool_call_leak_never_holds_back_normal_short_reply():
    # A short, ordinary reply must not get eaten by the holdback buffer
    # once the stream actually ends.
    out = run_filter(_filter_tool_call_leak, ["Hi"])
    assert out == "Hi"


def test_filters_compose_channel_then_tool_call():
    text = ('<|channel>thought<channel|>Let me check.'
            '<|tool_call>call:web_search{queries:[<|"|>x<|"|>]}<tool_call|>')
    out = run_filter(_filter_tool_call_leak, _filter_channel_tags(iter([text])))
    assert out == "Let me check."


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
