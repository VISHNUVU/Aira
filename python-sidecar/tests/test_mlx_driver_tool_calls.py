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
    tok.tool_call_start = "<|tool_call>"
    tok.tool_call_end = "<tool_call|>"
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


# ---- VLM-mode tool-calling (regression: this was silently dead for every
# real Gemma-4 checkpoint, since they're all unified/multimodal exports that
# load through mlx_vlm, not mlx_lm — confirmed live against the real app) ---

def _vlm_driver(chat_template=None, nested=False):
    tmp = tempfile.mkdtemp()
    from engine.mlx_driver import MLXDriver
    d = MLXDriver(os.path.join(tmp, "models"), os.path.join(tmp, "adapters"))
    d._vlm_mode = True

    class _FakeProcessor:
        pass

    proc = _FakeProcessor()
    if nested:
        class _FakeInnerTokenizer:
            pass
        inner = _FakeInnerTokenizer()
        inner.chat_template = chat_template
        proc.tokenizer = inner
    else:
        proc.chat_template = chat_template
    d._tokenizer = proc
    return d


def test_resolve_tool_calling_vlm_mode_detects_gemma4_template():
    d = _vlm_driver(chat_template="...<|tool_call>...<tool_call|>...")
    resolved = d._resolve_tool_calling()
    assert resolved is not None
    start, end, parser = resolved
    assert start == "<|tool_call>" and end == "<tool_call|>"
    assert parser.__name__ == "parse_tool_call"


def test_resolve_tool_calling_vlm_mode_checks_nested_tokenizer_attr():
    # mlx_vlm's processor sometimes only exposes chat_template via a nested
    # .tokenizer attribute rather than directly on the processor itself.
    d = _vlm_driver(chat_template="...<|tool_call>...<tool_call|>...", nested=True)
    resolved = d._resolve_tool_calling()
    assert resolved is not None


def test_resolve_tool_calling_vlm_mode_no_match_returns_none():
    d = _vlm_driver(chat_template="a plain template with no tool syntax at all")
    assert d._resolve_tool_calling() is None


def test_resolve_tool_calling_vlm_mode_missing_template_returns_none():
    d = _vlm_driver(chat_template=None)
    assert d._resolve_tool_calling() is None


def test_generate_vlm_forwards_tools_to_chat_template():
    from unittest.mock import patch, MagicMock
    d = _vlm_driver(chat_template="...<|tool_call>...<tool_call|>...")
    d._vlm_config = {}
    d._vlm_cache_state = None

    fake_resp = MagicMock(text="hi")
    tools = [{"type": "function", "function": {"name": "current_time"}}]
    with patch("mlx_vlm.prompt_utils.apply_chat_template") as mock_apply, \
         patch("mlx_vlm.stream_generate", return_value=iter([fake_resp])):
        mock_apply.return_value = "PROMPT"
        list(d._generate_vlm([{"role": "user", "content": "hi"}], max_tokens=10,
                             temperature=0.7, tools=tools, stream=False))
    call_kwargs = mock_apply.call_args.kwargs
    assert call_kwargs.get("tools") == tools


def test_generate_vlm_omits_tools_kwarg_when_none():
    from unittest.mock import patch, MagicMock
    d = _vlm_driver(chat_template="...<|tool_call>...<tool_call|>...")
    d._vlm_config = {}
    d._vlm_cache_state = None

    fake_resp = MagicMock(text="hi")
    with patch("mlx_vlm.prompt_utils.apply_chat_template") as mock_apply, \
         patch("mlx_vlm.stream_generate", return_value=iter([fake_resp])):
        mock_apply.return_value = "PROMPT"
        list(d._generate_vlm([{"role": "user", "content": "hi"}], max_tokens=10,
                             temperature=0.7, tools=None, stream=False))
    assert "tools" not in mock_apply.call_args.kwargs


# ---- real-model smoke test (opt-in, needs a real downloaded MLX VLM
# checkpoint + Metal) — the whole reason this section exists: every other
# test above mocks mlx_vlm entirely, which is exactly how the tool-calling
# loop shipped completely non-functional for every real Gemma-4 checkpoint
# despite passing all of them (confirmed live in production before this
# fix). This is the regression guard for that specific failure mode. -------

def test_real_vlm_tool_calling_end_to_end():
    if os.environ.get("ARIA_TEST_REAL_MLX_VLM") != "1":
        print("SKIP (set ARIA_TEST_REAL_MLX_VLM=1 to run against a real "
              "downloaded Gemma-4 VLM checkpoint, e.g. gemma-4-12b)")
        return
    from engine.mlx_driver import MLXDriver

    models_dir = os.path.expanduser("~/.aria/models")
    adapters_dir = os.path.expanduser("~/.aria/adapters")
    model_id = os.environ.get("ARIA_TEST_REAL_MLX_VLM_MODEL", "gemma-4-12b")
    d = MLXDriver(models_dir, adapters_dir)
    d.load(model_id)
    assert d._vlm_mode is True, (
        f"{model_id} did not load in VLM mode — this test needs a real "
        "unified/multimodal Gemma-4 checkpoint to guard against the actual bug"
    )

    tools = [{
        "type": "function",
        "function": {
            "name": "current_time",
            "description": "Get the current local date and time.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }]
    chunks = list(d.generate(
        [{"role": "user", "content": "Use your current_time tool to tell me "
                                     "the exact time right now."}],
        tools=tools, stream=False,
    ))
    tool_spans = [c for c in chunks if isinstance(c, ToolCallSpan)]
    assert tool_spans, (
        f"model never emitted a tool call for an explicit tool request — "
        f"got chunks: {chunks!r}. This is the exact failure mode this test "
        f"exists to catch."
    )
    calls = d.parse_tool_calls(tool_spans[0].raw_text, tools)
    assert any(c["name"] == "current_time" for c in calls), (
        f"captured a tool-call span but couldn't parse it: {tool_spans[0].raw_text!r}"
    )


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
