"""Tests for the tool / function-calling layer."""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from store import Store
from tools import (Tool, ToolRegistry, default_registry, make_current_time_tool,
                   make_shell_tool)


def add_tool(reg):
    def add(a: int, b: int) -> int:
        return a + b
    reg.register(Tool(
        name="add", description="add two ints",
        parameters={"type": "object",
                    "properties": {"a": {"type": "integer"},
                                   "b": {"type": "integer"}},
                    "required": ["a", "b"]},
        func=add))


def test_register_and_spec():
    reg = ToolRegistry(Store(":memory:"))
    add_tool(reg)
    specs = reg.specs()
    assert len(specs) == 1
    fn = specs[0]["function"]
    assert fn["name"] == "add"
    assert fn["parameters"]["required"] == ["a", "b"]


def test_dispatch_ok_and_logs():
    store = Store(":memory:")
    reg = ToolRegistry(store)
    add_tool(reg)
    r = reg.dispatch("add", {"a": 2, "b": 3}, turn_id="t1")
    assert r.ok and r.result == 5
    log = reg.call_log()
    assert len(log) == 1
    assert log[0]["tool_name"] == "add"
    assert log[0]["status"] == "ok"
    assert json.loads(log[0]["arguments"]) == {"a": 2, "b": 3}


def test_dispatch_unknown_tool():
    reg = ToolRegistry(Store(":memory:"))
    r = reg.dispatch("nope", {})
    assert not r.ok and "unknown tool" in r.error
    assert reg.call_log()[0]["status"] == "error"


def test_dispatch_bad_arguments():
    reg = ToolRegistry(Store(":memory:"))
    add_tool(reg)
    r = reg.dispatch("add", {"a": 1})       # missing b
    assert not r.ok
    assert reg.call_log()[0]["status"] == "error"


def test_dispatch_tool_raises():
    reg = ToolRegistry(Store(":memory:"))
    reg.register(Tool(name="boom", description="raises",
                      parameters={"type": "object", "properties": {}, "required": []},
                      func=lambda: (_ for _ in ()).throw(ValueError("kaboom"))))
    r = reg.dispatch("boom", {})
    assert not r.ok and "kaboom" in r.error


def test_disabled_tool_denied():
    reg = ToolRegistry(Store(":memory:"))
    add_tool(reg)
    reg.set_enabled("add", False)
    r = reg.dispatch("add", {"a": 1, "b": 2})
    assert not r.ok
    assert reg.call_log()[0]["status"] == "denied"
    # disabled tool excluded from specs by default
    assert reg.specs() == []
    assert len(reg.specs(enabled_only=False)) == 1


def test_shell_tool_off_by_default():
    reg = ToolRegistry(Store(":memory:"))
    reg.register(make_shell_tool())
    r = reg.dispatch("run_shell", {"command": "echo hi"})
    assert not r.ok            # denied because disabled
    # after explicit enable it runs
    reg.set_enabled("run_shell", True)
    r2 = reg.dispatch("run_shell", {"command": "echo hi"})
    assert r2.ok and r2.result["returncode"] == 0
    assert "hi" in r2.result["stdout"]


def test_file_search_tool_over_memory():
    # integration with the RAG memory module
    from engine import make_engine
    from memory import Memory, InMemoryVectorStore
    import tempfile
    from tools import make_file_search_tool
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(":memory:")
        eng = make_engine("fake", tmp + "/m", tmp + "/a", dim=64)
        eng.load("gemma-4-e4b")
        mem = Memory(store, eng, InMemoryVectorStore())
        mem.ingest_text("The capital of France is Paris.", source="geo.txt")
        reg = ToolRegistry(store)
        reg.register(make_file_search_tool(mem))
        r = reg.dispatch("file_search", {"query": "France capital", "k": 3})
        assert r.ok and len(r.result) >= 1
        assert r.result[0]["source"] == "geo.txt"


def test_current_time_tool():
    reg = ToolRegistry(Store(":memory:"))
    reg.register(make_current_time_tool())
    r = reg.dispatch("current_time", {})
    assert r.ok and isinstance(r.result, str) and len(r.result) > 0


def test_default_registry_has_shell_disabled():
    reg = default_registry(Store(":memory:"))
    shell = reg.get("run_shell")
    assert shell is not None and shell.enabled is False and shell.dangerous is True
    assert reg.get("current_time").enabled is True


def test_default_registry_has_web_search_disabled():
    reg = default_registry(Store(":memory:"))
    web_search = reg.get("web_search")
    assert web_search is not None
    assert web_search.enabled is False and web_search.dangerous is True


def test_default_registry_omits_image_generation_without_images_dir():
    reg = default_registry(Store(":memory:"))
    assert reg.get("image_generation") is None


def test_default_registry_has_image_generation_disabled_when_images_dir_given():
    reg = default_registry(Store(":memory:"), images_dir="/tmp/aria-test-images")
    tool = reg.get("image_generation")
    assert tool is not None
    assert tool.enabled is False and tool.dangerous is False


def test_enabled_state_survives_registry_recreation():
    # Regression: a user's enable/disable choice must survive the sidecar
    # process restarting (e.g. relaunching the app, or a fresh process after
    # an app update) — set_enabled() must persist to the store, and
    # register() must restore it, not just trust the hardcoded default.
    store = Store(":memory:")
    reg = default_registry(store)
    assert reg.get("web_search").enabled is False   # off by default
    reg.set_enabled("web_search", True)
    reg.set_enabled("current_time", False)          # on by default -> turn off

    # Simulate a new process: same on-disk store, brand-new registry.
    reg2 = default_registry(store)
    assert reg2.get("web_search").enabled is True
    assert reg2.get("current_time").enabled is False
    # A tool never explicitly toggled still falls back to its own default.
    assert reg2.get("file_search") is None  # no memory passed in this test
    assert reg2.get("run_shell").enabled is False


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
