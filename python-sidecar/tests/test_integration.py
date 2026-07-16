"""End-to-end integration test: the whole sidecar wired with a fake engine.

Exercises the real code paths the Tauri UI uses — via SidecarService methods
and via the HTTP _route dispatcher (no socket needed) — proving memory,
chat+RAG, tools, feedback->train->promote, adapters, and models all cooperate.
"""
import base64
import os
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app
from app import SidecarService, _route
from engine import ToolCallSpan
from eval_gate import EvalGate


class VersionScoreEvaluator:
    """Deterministic: score increases with adapter version number, so a newer
    adapter always beats an older one (and any adapter beats the base model).

    Lets us test promote/activate paths reproducibly — the real embedding
    evaluator is stochastic with the fake engine's random embeddings."""
    def score(self, engine, adapter_path, held_out):
        if adapter_path is None:
            return 0.5                      # base model
        ver = int(os.path.basename(adapter_path.rstrip("/")).lstrip("v"))
        return 0.5 + 0.1 * ver


def fresh_service(deterministic_gate=False):
    tmp = tempfile.mkdtemp()
    svc = SidecarService(engine_name="fake", home=tmp, use_lance=False, embed_dim=64)
    svc.load_model("gemma-4-e4b")
    if deterministic_gate:
        svc.trainer.gate = EvalGate(VersionScoreEvaluator(), min_improvement=0.0)
    return svc


def test_status_shape():
    svc = fresh_service()
    st = svc.status()
    for key in ("engine", "model", "capabilities", "memory", "feedback", "loaded"):
        assert key in st, key
    assert st["loaded"] is True
    assert st["capabilities"]["train"] is True   # fake engine can train


def test_memory_roundtrip_and_rag_chat():
    svc = fresh_service()
    r = svc.memory_add("The Eiffel Tower is located in Paris, France.", source="geo")
    assert r["chunks_added"] == 1
    chunks = svc.memory_list()
    assert len(chunks) == 1
    # chat pulls the context in
    out = svc.chat([{"role": "user", "content": "Where is the Eiffel Tower?"}])
    assert out["used_context"] is True
    assert isinstance(out["content"], str) and len(out["content"]) > 0
    # delete
    svc.memory_delete(chunks[0]["id"])
    assert len(svc.memory_list()) == 0


def test_tools_via_service():
    svc = fresh_service()
    names = [t["name"] for t in svc.tools_list()]
    assert "current_time" in names and "file_search" in names
    r = svc.tool_invoke("current_time", {})
    assert r["ok"] is True
    # shell is off by default -> invoke denied
    r2 = svc.tool_invoke("run_shell", {"command": "echo hi"})
    assert r2["ok"] is False
    # enable then works
    svc.tool_toggle("run_shell", True)
    r3 = svc.tool_invoke("run_shell", {"command": "echo hi"})
    assert r3["ok"] is True
    # every call logged
    assert len(svc.tool_calls()) >= 3


def test_feedback_train_promote_flow():
    svc = fresh_service(deterministic_gate=True)
    for i in range(8):
        svc.feedback_add(f"question {i}", f"the preferred answer {i}")
    summary = svc.train_now()
    assert summary["status"] == "promoted", summary
    adapters = svc.adapters_list()
    assert len(adapters) == 1 and adapters[0]["is_active"] == 1
    # runs recorded
    assert svc.training_runs()[0]["status"] == "promoted"


def test_adapter_activate_and_rollback():
    svc = fresh_service(deterministic_gate=True)
    # two rounds -> two adapters
    for i in range(8):
        svc.feedback_add(f"a q {i}", f"a ans {i}")
    svc.train_now()
    for i in range(8):
        svc.feedback_add(f"b q {i}", f"b ans {i}")
    svc.train_now()
    adapters = svc.adapters_list()
    assert len(adapters) == 2
    # activate the older one explicitly
    svc.adapter_activate("v0001")
    assert svc.registry.active()["id"] == "v0001"
    assert svc.engine.current_adapter.endswith("v0001")


def test_models_catalog():
    svc = fresh_service()
    m = svc.models_list()
    assert "gemma-4-12b" in m["catalog"]
    assert "gemma-4-e4b" in m["catalog"]
    # unknown download fails cleanly
    assert svc.model_download("nope")["ok"] is False


def test_http_route_dispatch():
    svc = fresh_service()
    assert _route(svc, "GET", "/status", {})[0] == 200
    assert _route(svc, "GET", "/nonexistent", {})[0] == 404
    assert _route(svc, "POST", "/memory", {})[0] == 400        # missing 'text'
    code, payload = _route(svc, "POST", "/memory",
                           {"text": "hi there", "source": "x"})
    assert code == 200 and payload["chunks_added"] >= 1
    # tool invoke through the router
    code, payload = _route(svc, "POST", "/tools/invoke",
                           {"name": "current_time", "arguments": {}})
    assert code == 200 and payload["ok"] is True


def test_memory_upload_txt_via_service():
    svc = fresh_service()
    b64 = base64.b64encode(b"a fact worth remembering").decode()
    r = svc.memory_upload("notes.txt", b64)
    assert r["chunks_added"] >= 1
    assert r["filename"] == "notes.txt"
    chunks = svc.memory_list()
    assert any("a fact worth remembering" in c["preview"] for c in chunks)


def test_memory_upload_rejects_bad_base64():
    svc = fresh_service()
    try:
        svc.memory_upload("notes.txt", "not-valid-base64!!!")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_memory_upload_route_returns_400_for_unsupported_type():
    svc = fresh_service()
    b64 = base64.b64encode(b"binary junk").decode()
    code, payload = _route(svc, "POST", "/memory/upload",
                           {"filename": "photo.png", "content_b64": b64})
    assert code == 400
    assert "unsupported" in payload["error"].lower()


def test_memory_upload_route_success():
    svc = fresh_service()
    b64 = base64.b64encode(b"uploaded via the http route").decode()
    code, payload = _route(svc, "POST", "/memory/upload",
                           {"filename": "doc.md", "content_b64": b64})
    assert code == 200
    assert payload["chunks_added"] >= 1


def test_chat_without_session_id_creates_and_returns_one():
    svc = fresh_service()
    r = svc.chat([{"role": "user", "content": "hello there"}])
    assert r["session_id"]
    sessions = svc.chat_history.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["id"] == r["session_id"]
    assert sessions[0]["title"] == "hello there"


def test_chat_persists_user_and_assistant_turns():
    svc = fresh_service()
    r = svc.chat([{"role": "user", "content": "hello there"}])
    msgs = svc.chat_history.get_messages(r["session_id"])
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "hello there"
    assert msgs[1]["content"] == r["content"]


def test_chat_reuses_passed_session_id_across_turns():
    svc = fresh_service()
    r1 = svc.chat([{"role": "user", "content": "first"}])
    sid = r1["session_id"]
    r2 = svc.chat([{"role": "user", "content": "first"},
                   {"role": "assistant", "content": r1["content"]},
                   {"role": "user", "content": "second"}], session_id=sid)
    assert r2["session_id"] == sid
    msgs = svc.chat_history.get_messages(sid)
    assert len(msgs) == 4
    assert svc.chat_history.list_sessions()[0]["title"] == "first"  # unchanged


def test_chat_stream_persists_full_assembled_reply():
    svc = fresh_service()
    events = list(svc.chat_stream([{"role": "user", "content": "hi"}]))
    done = next(e for e in events if e.get("done"))
    sid = done["session_id"]
    msgs = svc.chat_history.get_messages(sid)
    assert msgs[0]["content"] == "hi"
    full_reply = "".join(e["delta"] for e in events if "delta" in e)
    assert msgs[1]["content"] == full_reply.strip()  # add_message() strips content


def test_sessions_routes_via_http_dispatch():
    svc = fresh_service()
    r1 = svc.chat([{"role": "user", "content": "route test"}])
    sid = r1["session_id"]

    code, payload = _route(svc, "GET", "/sessions", {})
    assert code == 200 and any(s["id"] == sid for s in payload)

    code, payload = _route(svc, "GET", "/sessions/messages", {"session_id": sid})
    assert code == 200 and len(payload) == 2

    code, payload = _route(svc, "POST", "/sessions/rename",
                           {"session_id": sid, "title": "Renamed"})
    assert code == 200 and payload["title"] == "Renamed"

    code, payload = _route(svc, "POST", "/sessions/delete", {"session_id": sid})
    assert code == 200 and payload["ok"] is True
    assert svc.chat_history.get_session(sid) is None


def test_persona_defaults_to_built_in():
    svc = fresh_service()
    p = svc.get_persona()
    assert p["is_custom"] is False
    assert p["prompt"] == svc.PERSONA_PROMPT == p["default"]


def test_persona_custom_prompt_used_in_chat():
    svc = fresh_service()
    svc.set_persona("Speak only in rhymes.")
    p = svc.get_persona()
    assert p["is_custom"] is True and p["prompt"] == "Speak only in rhymes."
    msgs, *_ = svc._prepare_chat([{"role": "user", "content": "hi"}],
                                 use_memory=False, use_tools=False)
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == "Speak only in rhymes."


def test_persona_reset_reverts_to_default():
    svc = fresh_service()
    svc.set_persona("Custom prompt")
    svc.reset_persona()
    p = svc.get_persona()
    assert p["is_custom"] is False and p["prompt"] == svc.PERSONA_PROMPT


def test_persona_rejects_empty_prompt():
    svc = fresh_service()
    try:
        svc.set_persona("   ")
        assert False, "should have raised"
    except ValueError:
        pass


def test_persona_rejects_oversized_prompt():
    svc = fresh_service()
    try:
        svc.set_persona("x" * (svc.PERSONA_MAX_CHARS + 1))
        assert False, "should have raised"
    except ValueError:
        pass
    # boundary: exactly at the cap must succeed
    svc.set_persona("x" * svc.PERSONA_MAX_CHARS)
    assert svc.get_persona()["is_custom"] is True


def test_web_search_disabled_by_default_never_triggers():
    svc = fresh_service()
    with patch("app.extract_search_query") as mock_extract:
        msgs, *_, used_search = svc._prepare_chat(
            [{"role": "user", "content": "search for the weather in tokyo"}],
            use_memory=True, use_tools=True)
    assert used_search is None
    mock_extract.assert_not_called()  # short-circuits on tool.enabled before even checking


def test_web_search_triggers_when_enabled():
    svc = fresh_service()
    svc.tools.set_enabled("web_search", True)
    fake_results = [{"title": "Tokyo Weather", "url": "https://example.com",
                     "snippet": "Sunny, 22C"}]
    with patch("app.extract_search_query", return_value="weather in tokyo"), \
         patch.object(svc.tools.get("web_search"), "func", return_value=fake_results):
        msgs, context, auto_saved, used_skills, used_search = svc._prepare_chat(
            [{"role": "user", "content": "search for the weather in tokyo"}],
            use_memory=True, use_tools=True)
    assert used_search == {"query": "weather in tokyo", "results": fake_results}
    # results actually reached the prompt, not just the return value
    assert any("Tokyo Weather" in m.get("content", "") for m in msgs)


def test_web_search_ignores_ordinary_chat_even_when_enabled():
    svc = fresh_service()
    svc.tools.set_enabled("web_search", True)
    msgs, *_, used_search = svc._prepare_chat(
        [{"role": "user", "content": "how's it going"}],
        use_memory=True, use_tools=True)
    assert used_search is None


def test_web_search_respects_use_tools_flag():
    svc = fresh_service()
    svc.tools.set_enabled("web_search", True)
    msgs, *_, used_search = svc._prepare_chat(
        [{"role": "user", "content": "search for the weather in tokyo"}],
        use_memory=True, use_tools=False)
    assert used_search is None


def _run_thread_synchronously(target, daemon=True):
    """Fake threading.Thread that runs its target immediately in the
    calling thread and exposes a no-op .start() — lets tests observe
    image_generate()'s background work without a real race or sleep."""
    class _Sync:
        def start(self_inner):
            target()
    return _Sync()


def test_image_generation_disabled_by_default_never_triggers():
    svc = fresh_service()
    with patch("app.extract_image_prompt") as mock_extract:
        r = svc.chat([{"role": "user", "content": "generate an image of a cat"}])
    assert "image_job" not in r
    mock_extract.assert_not_called()  # short-circuits on tool.enabled before even checking


def test_image_generation_triggers_when_enabled():
    svc = fresh_service()
    svc.tools.set_enabled("image_generation", True)
    fake_result = {"path": "/tmp/x.png", "filename": "x.png", "seconds": 1.0, "seed": 1}
    with patch("app.image_gen_available", return_value=True), \
         patch("app.run_image_generation", return_value=fake_result), \
         patch("app.threading.Thread", side_effect=_run_thread_synchronously):
        r = svc.chat([{"role": "user", "content": "generate an image of a cat"}])
    assert r["image_job"]["ok"] is True
    assert "cat" in r["content"]
    progress = svc.image_generate_progress()
    assert progress["status"] == "done"
    assert progress["result"]["url"] == "/images/file?filename=x.png"


def test_image_generation_persists_final_result_to_chat_history():
    # Regression test: chat_history.add_message() used to drop image_job
    # entirely, so reopening a past chat could never show the generated
    # image — only the static "Generating..." placeholder text, forever.
    svc = fresh_service()
    svc.tools.set_enabled("image_generation", True)
    fake_result = {"path": "/tmp/x.png", "filename": "x.png", "seconds": 1.0, "seed": 1}
    with patch("app.image_gen_available", return_value=True), \
         patch("app.run_image_generation", return_value=fake_result), \
         patch("app.threading.Thread", side_effect=_run_thread_synchronously):
        r = svc.chat([{"role": "user", "content": "generate an image of a cat"}])
    msgs = svc.chat_history.get_messages(r["session_id"])
    assistant_msg = [m for m in msgs if m["role"] == "assistant"][0]
    assert assistant_msg["image_job"]["status"] == "done"
    assert assistant_msg["image_job"]["result"]["url"] == "/images/file?filename=x.png"
    # Regression: the persisted job dict once lacked an "ok" key entirely
    # (only the transient in-flight dict had it) — the frontend's
    # renderImageJob() checks `!imageJob.ok` first on every reload, so a
    # missing key read as falsy and showed a false "couldn't start" error
    # even though generation had actually succeeded.
    assert assistant_msg["image_job"]["ok"] is True


def test_image_generation_ignores_ordinary_chat_even_when_enabled():
    svc = fresh_service()
    svc.tools.set_enabled("image_generation", True)
    r = svc.chat([{"role": "user", "content": "how's it going"}])
    assert "image_job" not in r


def test_image_generation_respects_use_tools_flag():
    svc = fresh_service()
    svc.tools.set_enabled("image_generation", True)
    r = svc.chat([{"role": "user", "content": "generate an image of a cat"}], use_tools=False)
    assert "image_job" not in r


def test_image_generation_reports_unavailable_build_cleanly():
    svc = fresh_service()
    svc.tools.set_enabled("image_generation", True)
    with patch("app.image_gen_available", return_value=False):
        r = svc.chat([{"role": "user", "content": "generate an image of a cat"}])
    assert r["image_job"]["ok"] is False
    assert "isn't available" in r["content"].lower()


def test_image_generate_route_via_http_dispatch():
    svc = fresh_service()
    fake_result = {"path": "/tmp/x.png", "filename": "x.png", "seconds": 1.0, "seed": 1}
    with patch("app.image_gen_available", return_value=True), \
         patch("app.run_image_generation", return_value=fake_result), \
         patch("app.threading.Thread", side_effect=_run_thread_synchronously):
        code, payload = _route(svc, "POST", "/images/generate", {"prompt": "a dog"})
    assert code == 200 and payload["ok"] is True
    code, payload = _route(svc, "GET", "/images/progress", {})
    assert code == 200 and payload["status"] == "done"
    code, payload = _route(svc, "GET", "/images", {})
    assert code == 200


def test_tool_loop_dispatches_a_model_requested_tool_call():
    svc = fresh_service()
    calls = [{"name": "current_time", "arguments": {}}]

    def round1():
        yield ToolCallSpan(raw_text="<|tool_call>call:current_time{}<tool_call|>")

    def round2():
        yield "It is 3pm."

    rounds = iter([round1, round2])

    def dispatch_round(*a, **kw):
        return next(rounds)()

    with patch.object(svc.engine, "generate", side_effect=dispatch_round), \
         patch.object(svc.engine, "parse_tool_calls", return_value=calls):
        r = svc.chat([{"role": "user", "content": "what time is it"}])
    assert r["used_tools"] == ["current_time"]
    assert "3pm" in r["content"]
    # dispatch() was real (not mocked) -> logged in the real call log
    assert any(c["tool_name"] == "current_time" for c in svc.tool_calls())


def test_tool_loop_appends_tool_call_and_result_messages_before_next_round():
    svc = fresh_service()
    calls = [{"name": "current_time", "arguments": {}}]
    captured = []

    def round1(msgs):
        captured.append(msgs)
        yield ToolCallSpan(raw_text="<|tool_call>call:current_time{}<tool_call|>")

    def round2(msgs):
        captured.append(msgs)
        yield "done"

    rounds = iter([round1, round2])

    def dispatch_round(msgs, *a, **kw):
        return next(rounds)(msgs)

    with patch.object(svc.engine, "generate", side_effect=dispatch_round), \
         patch.object(svc.engine, "parse_tool_calls", return_value=calls):
        svc.chat([{"role": "user", "content": "what time is it"}])
    # both calls share the same mutated msgs list -> inspect it after the fact
    msgs_after = captured[-1]
    assert msgs_after[-2]["role"] == "assistant"
    assert msgs_after[-2]["tool_calls"][0]["function"]["name"] == "current_time"
    assert msgs_after[-1]["role"] == "tool"
    assert msgs_after[-1]["tool_call_id"] == msgs_after[-2]["tool_calls"][0]["id"]


def test_tool_loop_stops_when_tool_call_is_unparseable():
    svc = fresh_service()

    def round1(*a, **kw):
        yield ToolCallSpan(raw_text="garbage")

    with patch.object(svc.engine, "generate", side_effect=round1) as mock_generate, \
         patch.object(svc.engine, "parse_tool_calls", return_value=[]):
        r = svc.chat([{"role": "user", "content": "hi"}])
        assert mock_generate.call_count == 1
    assert r["used_tools"] == []


def test_tool_loop_respects_max_iterations_cap():
    svc = fresh_service()
    calls = [{"name": "current_time", "arguments": {}}]

    def always_calls_tool(*a, **kw):
        yield ToolCallSpan(raw_text="<|tool_call>call:current_time{}<tool_call|>")

    with patch.object(svc.engine, "generate", side_effect=always_calls_tool) as mock_generate, \
         patch.object(svc.engine, "parse_tool_calls", return_value=calls):
        r = svc.chat([{"role": "user", "content": "loop forever"}])
        assert mock_generate.call_count == SidecarService.MAX_TOOL_ITERATIONS
    assert r["used_tools"] == ["current_time"] * SidecarService.MAX_TOOL_ITERATIONS


def test_tool_loop_respects_use_tools_flag():
    svc = fresh_service()
    with patch.object(svc.engine, "generate", wraps=svc.engine.generate) as spy:
        svc.chat([{"role": "user", "content": "hi"}], use_tools=False)
    _, kwargs = spy.call_args
    assert kwargs["tools"] is None


def test_tool_loop_respects_cross_session_call_budget():
    # A model that develops a habit of calling a tool every turn has no
    # ceiling other than this — MAX_TOOL_ITERATIONS only caps a single turn.
    # ensure_session() only reuses a passed session_id if it already exists
    # (chat_history.py) — so the real id has to come from the first call's
    # response, not an invented literal, or each "same session" call below
    # would silently land in its own brand-new session instead.
    svc = fresh_service()
    calls = [{"name": "current_time", "arguments": {}}]

    def always_calls_tool(*a, **kw):
        yield ToolCallSpan(raw_text="<|tool_call>call:current_time{}<tool_call|>")

    with patch.object(svc.engine, "generate", side_effect=always_calls_tool), \
         patch.object(svc.engine, "parse_tool_calls", return_value=calls):
        r0 = svc.chat([{"role": "user", "content": "what time is it"}])
        session_id = r0["session_id"]
        # MAX_TOOL_ITERATIONS(4) * 5 calls == MAX_TOOL_CALLS_PER_SESSION(20)
        # exactly — the budget should be fully spent, not yet exceeded, here.
        for _ in range(4):
            svc.chat([{"role": "user", "content": "what time is it"}],
                    session_id=session_id)
    assert svc._session_tool_call_counts[session_id] == SidecarService.MAX_TOOL_CALLS_PER_SESSION

    # The next call in the SAME session must not offer tools at all anymore.
    with patch.object(svc.engine, "generate", wraps=svc.engine.generate) as spy, \
         patch.object(svc.engine, "parse_tool_calls", return_value=calls):
        r = svc.chat([{"role": "user", "content": "what time is it"}],
                    session_id=session_id)
    _, kwargs = spy.call_args
    assert kwargs["tools"] is None
    assert r["used_tools"] == []


def test_tool_loop_session_budgets_are_independent():
    svc = fresh_service()
    calls = [{"name": "current_time", "arguments": {}}]

    def always_calls_tool(*a, **kw):
        yield ToolCallSpan(raw_text="<|tool_call>call:current_time{}<tool_call|>")

    with patch.object(svc.engine, "generate", side_effect=always_calls_tool), \
         patch.object(svc.engine, "parse_tool_calls", return_value=calls):
        r0 = svc.chat([{"role": "user", "content": "what time is it"}])
        session_a = r0["session_id"]
        for _ in range(4):
            svc.chat([{"role": "user", "content": "what time is it"}],
                    session_id=session_a)

    # A different (brand-new) session's budget must be untouched by
    # session_a's usage.
    with patch.object(svc.engine, "generate", wraps=svc.engine.generate) as spy, \
         patch.object(svc.engine, "parse_tool_calls", return_value=calls):
        svc.chat([{"role": "user", "content": "hi"}])
    _, kwargs = spy.call_args
    assert kwargs["tools"] is not None


def test_load_model_remembers_last_model():
    svc = fresh_service()  # already loads gemma-4-e4b in fresh_service()
    assert svc.store.get_meta("last_model_id") == "gemma-4-e4b"


def test_auto_load_last_model_noop_when_none_remembered():
    tmp = tempfile.mkdtemp()
    svc = SidecarService(engine_name="fake", home=tmp, use_lance=False, embed_dim=64)
    assert svc.status()["loaded"] is False
    svc.auto_load_last_model()
    assert svc.status()["loaded"] is False


def test_auto_load_last_model_noop_when_weights_missing():
    tmp = tempfile.mkdtemp()
    svc = SidecarService(engine_name="fake", home=tmp, use_lance=False, embed_dim=64)
    svc.store.set_meta("last_model_id", "gemma-4-e4b")  # remembered, but never downloaded here
    svc.auto_load_last_model()
    assert svc.status()["loaded"] is False


def test_auto_load_last_model_reloads_downloaded_model():
    tmp = tempfile.mkdtemp()
    svc = SidecarService(engine_name="fake", home=tmp, use_lance=False, embed_dim=64)
    os.makedirs(os.path.join(svc.models_dir, "gemma-4-e4b"), exist_ok=True)
    svc.store.set_meta("last_model_id", "gemma-4-e4b")
    svc.auto_load_last_model()
    st = svc.status()
    assert st["loaded"] is True and st["model"] == "gemma-4-e4b"


def _fake_check_result(version="9.9.9"):
    return {
        "enabled": True, "ok": True, "current_version": "0.1.0",
        "latest_version": version, "update_available": True, "notes": "",
        "published_at": None, "asset_name": "Aria_9.9.9_aarch64.dmg",
        "asset_url": "http://example.com/Aria_9.9.9_aarch64.dmg",
        "asset_size": 123, "asset_sha256": "deadbeef",
    }


def test_update_auto_pull_noop_when_up_to_date():
    svc = fresh_service()
    with patch.object(svc.updater, "check", return_value={"enabled": True, "ok": True,
                                                           "update_available": False}):
        svc.update_auto_pull()
    assert svc.update_auto_status()["phase"] == "idle"


def test_update_auto_pull_reports_write_error_without_downloading():
    svc = fresh_service()
    with patch.object(svc.updater, "check", return_value=_fake_check_result()), \
         patch.object(svc.updater, "check_target_writable", return_value="no permission"), \
         patch.object(svc.updater, "download") as mock_download:
        svc.update_auto_pull()
    mock_download.assert_not_called()
    status = svc.update_auto_status()
    assert status["phase"] == "error"
    assert status["error"] == "no permission"


def test_update_auto_pull_downloads_and_installs_silently():
    svc = fresh_service()
    with patch.object(svc.updater, "check", return_value=_fake_check_result()), \
         patch.object(svc.updater, "check_target_writable", return_value=None), \
         patch.object(svc.updater, "download", return_value="/tmp/fake.dmg") as mock_download, \
         patch.object(svc.updater, "install_dmg") as mock_install, \
         patch("app.os.remove") as mock_remove:
        svc.update_auto_pull()
        for _ in range(200):
            if svc.update_auto_status()["phase"] in ("ready", "error"):
                break
            import time as _t; _t.sleep(0.01)
    status = svc.update_auto_status()
    assert status["phase"] == "ready", status
    assert status["version"] == "9.9.9"
    mock_download.assert_called_once()
    mock_install.assert_called_once_with("/tmp/fake.dmg")
    mock_remove.assert_called_once_with("/tmp/fake.dmg")


def test_update_auto_pull_does_not_redownload_same_version_twice():
    svc = fresh_service()
    svc._auto_update = {"phase": "ready", "version": "9.9.9", "error": None}
    with patch.object(svc.updater, "check", return_value=_fake_check_result()), \
         patch.object(svc.updater, "download") as mock_download:
        svc.update_auto_pull()
    mock_download.assert_not_called()


def test_update_relaunch_requires_ready_phase():
    svc = fresh_service()
    r = svc.update_relaunch()
    assert r["ok"] is False


def test_update_relaunch_opens_app_when_ready():
    svc = fresh_service()
    svc._auto_update = {"phase": "ready", "version": "9.9.9", "error": None}
    with patch("app.subprocess.Popen") as mock_popen:
        r = svc.update_relaunch()
    assert r["ok"] is True
    mock_popen.assert_called_once_with(["open", "-n", "/Applications/Aria.app"])


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
