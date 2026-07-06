"""End-to-end integration test: the whole sidecar wired with a fake engine.

Exercises the real code paths the Tauri UI uses — via SidecarService methods
and via the HTTP _route dispatcher (no socket needed) — proving memory,
chat+RAG, tools, feedback->train->promote, adapters, and models all cooperate.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app
from app import SidecarService, _route
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
