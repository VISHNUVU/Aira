"""Tests for LlamaCppDriver — the Windows/Linux inference backend.

llama-cpp-python itself runs fine on macOS (CPU backend), so most of this is
tested in isolation the same way test_mlx_driver_training.py mocks runpy: no
real GGUF weights needed for the unit-level tests. One real smoke test at the
bottom is guarded behind ARIA_TEST_REAL_GGUF=1 (off by default, like this
repo's other network-touching tests) — it downloads a tiny real GGUF and
proves load()/generate()/embed() work end-to-end, independent of ever
touching an actual Windows machine.
"""
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.base import EngineError, ToolCallSpan
from engine.llamacpp_driver import LlamaCppDriver


def fresh_driver(dim: int = 8) -> LlamaCppDriver:
    tmp = tempfile.mkdtemp()
    return LlamaCppDriver(os.path.join(tmp, "models"), os.path.join(tmp, "adapters"), dim=dim)


# ---- _resolve_gguf_path ----------------------------------------------------

def test_resolve_gguf_path_direct_file():
    d = fresh_driver()
    with tempfile.NamedTemporaryFile(suffix=".gguf") as f:
        assert d._resolve_gguf_path(f.name) == f.name


def test_resolve_gguf_path_model_dir():
    d = fresh_driver()
    model_dir = os.path.join(d.models_dir, "some-model")
    os.makedirs(model_dir)
    gguf_path = os.path.join(model_dir, "weights.gguf")
    open(gguf_path, "w").close()
    assert d._resolve_gguf_path("some-model") == gguf_path


def test_resolve_gguf_path_missing_raises():
    d = fresh_driver()
    try:
        d._resolve_gguf_path("nope")
        assert False, "expected EngineError"
    except EngineError:
        pass


def test_resolve_gguf_path_empty_dir_raises():
    d = fresh_driver()
    os.makedirs(os.path.join(d.models_dir, "empty-model"))
    try:
        d._resolve_gguf_path("empty-model")
        assert False, "expected EngineError"
    except EngineError:
        pass


# ---- parse_tool_calls -------------------------------------------------------

def test_parse_tool_calls_single():
    d = fresh_driver()
    calls = [{"id": "call_1", "type": "function",
              "function": {"name": "current_time", "arguments": "{}"}}]
    assert d.parse_tool_calls(json.dumps(calls)) == [{"name": "current_time", "arguments": {}}]


def test_parse_tool_calls_multiple():
    d = fresh_driver()
    calls = [
        {"id": "call_1", "type": "function",
         "function": {"name": "current_time", "arguments": "{}"}},
        {"id": "call_2", "type": "function",
         "function": {"name": "file_search", "arguments": '{"query": "paris"}'}},
    ]
    result = d.parse_tool_calls(json.dumps(calls))
    assert len(result) == 2
    assert result[1] == {"name": "file_search", "arguments": {"query": "paris"}}


def test_parse_tool_calls_malformed_returns_empty():
    d = fresh_driver()
    assert d.parse_tool_calls("not json at all") == []
    assert d.parse_tool_calls(json.dumps({"not": "a list"})) == []


# ---- _fit_dim ---------------------------------------------------------------

def test_fit_dim_exact():
    d = fresh_driver(dim=4)
    assert d._fit_dim([1.0, 2.0, 3.0, 4.0]) == [1.0, 2.0, 3.0, 4.0]


def test_fit_dim_truncates():
    d = fresh_driver(dim=2)
    assert d._fit_dim([1.0, 2.0, 3.0]) == [1.0, 2.0]


def test_fit_dim_pads():
    d = fresh_driver(dim=4)
    assert d._fit_dim([1.0, 2.0]) == [1.0, 2.0, 0.0, 0.0]


# ---- capabilities / train_lora ----------------------------------------------

def test_capabilities_no_training():
    d = fresh_driver()
    caps = d.capabilities
    assert caps.name == "llamacpp"
    assert caps.can_generate is True
    assert caps.can_embed is True
    assert caps.can_train is False


def test_train_lora_returns_informative_failure():
    d = fresh_driver()
    r = d.train_lora("dataset.jsonl", "/tmp/out", config=None)
    assert r.ok is False and "cannot train" in r.error


# ---- generate() — mocked llama_cpp.Llama ------------------------------------

def _model_file(models_dir: str, model_id: str) -> str:
    model_dir = os.path.join(models_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "weights.gguf")
    open(path, "w").close()
    return path


def test_generate_streams_plain_text_without_tools():
    d = fresh_driver()
    _model_file(d.models_dir, "m")
    fake_llm = MagicMock()
    fake_llm.create_chat_completion.return_value = iter([
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {}}]},
    ])
    with patch("llama_cpp.Llama", return_value=fake_llm):
        d.load("m")
    out = list(d.generate([{"role": "user", "content": "hi"}], tools=None, stream=True))
    assert out == ["Hel", "lo"]
    fake_llm.create_chat_completion.assert_called_once()
    assert "tools" not in fake_llm.create_chat_completion.call_args.kwargs


def test_generate_non_streaming_without_tools():
    d = fresh_driver()
    _model_file(d.models_dir, "m")
    fake_llm = MagicMock()
    fake_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "hello there"}}]
    }
    with patch("llama_cpp.Llama", return_value=fake_llm):
        d.load("m")
    out = list(d.generate([{"role": "user", "content": "hi"}], tools=None, stream=False))
    assert out == ["hello there"]


def test_generate_with_tools_yields_tool_call_span():
    d = fresh_driver()
    _model_file(d.models_dir, "m")
    fake_llm = MagicMock()
    tool_calls = [{"id": "call_1", "type": "function",
                   "function": {"name": "current_time", "arguments": "{}"}}]
    fake_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "", "tool_calls": tool_calls}}]
    }
    with patch("llama_cpp.Llama", return_value=fake_llm):
        d.load("m")
    tools = [{"type": "function", "function": {"name": "current_time"}}]
    out = list(d.generate([{"role": "user", "content": "what time is it"}], tools=tools))
    assert len(out) == 1
    assert isinstance(out[0], ToolCallSpan)
    assert d.parse_tool_calls(out[0].raw_text) == [{"name": "current_time", "arguments": {}}]
    call_kwargs = fake_llm.create_chat_completion.call_args.kwargs
    assert call_kwargs["tools"] == tools
    assert call_kwargs["stream"] is False


def test_generate_with_tools_yields_text_then_span_when_both_present():
    d = fresh_driver()
    _model_file(d.models_dir, "m")
    fake_llm = MagicMock()
    tool_calls = [{"id": "call_1", "type": "function",
                   "function": {"name": "current_time", "arguments": "{}"}}]
    fake_llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Let me check.", "tool_calls": tool_calls}}]
    }
    with patch("llama_cpp.Llama", return_value=fake_llm):
        d.load("m")
    out = list(d.generate([{"role": "user", "content": "hi"}],
                          tools=[{"type": "function", "function": {"name": "current_time"}}]))
    assert out[0] == "Let me check."
    assert isinstance(out[1], ToolCallSpan)


def test_generate_raises_if_no_model_loaded():
    d = fresh_driver()
    try:
        list(d.generate([{"role": "user", "content": "hi"}]))
        assert False, "expected EngineError"
    except EngineError:
        pass


# ---- embed() — mocked embedder ----------------------------------------------

def test_embed_uses_separate_embedder_and_fits_dim():
    d = fresh_driver(dim=3)
    _model_file(d.models_dir, d._embedding_model_id)
    fake_embedder = MagicMock()
    fake_embedder.create_embedding.return_value = {
        "data": [{"embedding": [1.0, 2.0]}, {"embedding": [1.0, 2.0, 3.0, 4.0]}]
    }
    with patch("llama_cpp.Llama", return_value=fake_embedder):
        vecs = d.embed(["a", "b"])
    assert vecs[0] == [1.0, 2.0, 0.0]     # padded to dim=3
    assert vecs[1] == [1.0, 2.0, 3.0]     # truncated to dim=3


def test_embed_collapses_per_token_vectors_with_mean_pool_shape():
    # Some GGUF pooling configs return one vector per token instead of one
    # pooled vector per input — this driver takes the first row rather than
    # failing, matching the documented fallback in embed()'s docstring.
    d = fresh_driver(dim=2)
    _model_file(d.models_dir, d._embedding_model_id)
    fake_embedder = MagicMock()
    fake_embedder.create_embedding.return_value = {
        "data": [{"embedding": [[1.0, 2.0], [3.0, 4.0]]}]
    }
    with patch("llama_cpp.Llama", return_value=fake_embedder):
        vecs = d.embed(["a"])
    assert vecs == [[1.0, 2.0]]


# ---- set_adapter() -----------------------------------------------------------

def test_set_adapter_reloads_model():
    d = fresh_driver()
    _model_file(d.models_dir, "m")
    with patch("llama_cpp.Llama", return_value=MagicMock()):
        d.load("m")
        d.set_adapter("/path/to/adapter")
    assert d._adapter_path == "/path/to/adapter"


def test_set_adapter_without_load_raises():
    d = fresh_driver()
    try:
        d.set_adapter("/path/to/adapter")
        assert False, "expected EngineError"
    except EngineError:
        pass


# ---- real smoke test (opt-in, needs network + a real GGUF download) --------

def test_real_gguf_end_to_end():
    if os.environ.get("ARIA_TEST_REAL_GGUF") != "1":
        print("SKIP (set ARIA_TEST_REAL_GGUF=1 to run against a real GGUF download)")
        return
    d = fresh_driver(dim=768)
    prog = {}
    d.download("gemma-4-e2b-gguf", "unsloth/gemma-4-E2B-it-GGUF",
              "gemma-4-E2B-it-Q4_K_M.gguf", progress_cb=lambda a, b: prog.update(a=a, b=b))
    d.load("gemma-4-e2b-gguf")
    out = "".join(d.generate([{"role": "user", "content": "Reply with exactly: ready"}],
                             tools=None, stream=False))
    assert isinstance(out, str) and len(out) > 0
    d.download("nomic-embed-text-v1.5", "nomic-ai/nomic-embed-text-v1.5-GGUF",
              "nomic-embed-text-v1.5.Q4_K_M.gguf")
    vecs = d.embed(["hello world"])
    assert len(vecs) == 1 and len(vecs[0]) == 768


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
