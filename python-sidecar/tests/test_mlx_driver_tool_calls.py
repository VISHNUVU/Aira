"""Tests for the native tool-calling loop's engine-layer pieces:
_split_tool_call (a pure generator, no mlx_lm needed, same style as
test_mlx_driver_filters.py) and MLXDriver.parse_tool_calls() (needs an
MLXDriver instance, but only touches a stubbed self._tokenizer — no real
mlx/mlx_lm/model load required).
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.base import ToolCallSpan
from engine.mlx_driver import MLXDriver, _split_tool_call

TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"


def run_split(chunks):
    return list(_split_tool_call(iter(chunks), TOOL_CALL_START, TOOL_CALL_END))


def joined_text(out):
    """Concatenate the plain-str parts of a _split_tool_call() result.

    A holdback buffer (_TOOL_CALL_HOLDBACK) means ordinary text can arrive
    split across multiple yielded chunks near the end of the stream — that's
    expected, not a bug, so tests compare joined text rather than exact
    chunk boundaries.
    """
    return "".join(p for p in out if isinstance(p, str))


def test_split_tool_call_streams_ordinary_text_unchanged():
    out = run_split(["Just a normal reply, nothing to see here."])
    assert all(isinstance(p, str) for p in out)
    assert joined_text(out) == "Just a normal reply, nothing to see here."


def test_split_tool_call_captures_complete_call_after_preamble():
    call = '<|tool_call>call:current_time{}<tool_call|>'
    out = run_split([f"Let me check. {call}"])
    assert isinstance(out[-1], ToolCallSpan)
    assert out[-1].raw_text == call
    assert joined_text(out) == "Let me check. "


def test_split_tool_call_captures_call_with_no_preamble():
    call = '<|tool_call>call:current_time{}<tool_call|>'
    out = run_split([call])
    assert len(out) == 1
    assert isinstance(out[0], ToolCallSpan)
    assert out[0].raw_text == call


def test_split_tool_call_drops_incomplete_call_silently():
    # No closing tag ever arrives (e.g. max_tokens hit mid-call) — matches
    # the existing truncating filter's fallback for this same situation.
    out = run_split(["some reply", "<|tool_call>call:current_time{"])
    assert out == ["some reply"]


def test_split_tool_call_handles_tag_split_across_chunks():
    call_start_split = ["some reply<|tool_", 'call>call:current_time{}<tool_call|>']
    out = run_split(call_start_split)
    assert isinstance(out[-1], ToolCallSpan)
    assert out[-1].raw_text == "<|tool_call>call:current_time{}<tool_call|>"
    assert joined_text(out) == "some reply"


def test_split_tool_call_never_holds_back_normal_short_reply():
    out = run_split(["Hi"])
    assert out == ["Hi"]


def _fake_driver_with_tokenizer(tool_parser=None, has_tool_calling=True):
    tmp = tempfile.mkdtemp()
    d = MLXDriver(os.path.join(tmp, "models"), os.path.join(tmp, "adapters"))

    class _FakeTokenizer:
        pass

    tok = _FakeTokenizer()
    tok.has_tool_calling = has_tool_calling
    tok.tool_parser = tool_parser
    d._tokenizer = tok
    return d


def test_parse_tool_calls_single_call():
    from mlx_lm.tool_parsers import gemma4
    d = _fake_driver_with_tokenizer(tool_parser=gemma4.parse_tool_call)
    text = '<|tool_call>call:current_time{}<tool_call|>'
    result = d.parse_tool_calls(text)
    assert result == [{"name": "current_time", "arguments": {}}]


def test_parse_tool_calls_multiple_calls_returns_a_list():
    from mlx_lm.tool_parsers import gemma4
    d = _fake_driver_with_tokenizer(tool_parser=gemma4.parse_tool_call)
    text = ('<|tool_call>call:current_time{}<tool_call|>'
            '<|tool_call>call:file_search{query:<|"|>test<|"|>}<tool_call|>')
    result = d.parse_tool_calls(text)
    assert len(result) == 2
    assert result[0]["name"] == "current_time"
    assert result[1] == {"name": "file_search", "arguments": {"query": "test"}}


def test_parse_tool_calls_malformed_text_returns_empty_list():
    from mlx_lm.tool_parsers import gemma4
    d = _fake_driver_with_tokenizer(tool_parser=gemma4.parse_tool_call)
    # No matches at all — real parse_tool_call raises ValueError here rather
    # than returning an empty result; must be normalized to [].
    result = d.parse_tool_calls("just some ordinary text, no tool call at all")
    assert result == []


def test_parse_tool_calls_no_capability_returns_empty_list():
    # No tool_parser attached (has_tool_calling False) — e.g. a non-Gemma-4
    # checkpoint, or VLM mode. Must not raise.
    d = _fake_driver_with_tokenizer(tool_parser=None, has_tool_calling=False)
    result = d.parse_tool_calls('<|tool_call>call:current_time{}<tool_call|>')
    assert result == []


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
