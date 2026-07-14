"""Tool / function-calling layer.

A registry of tools the model can call via Gemma 4's native function calling,
plus dispatch that logs every invocation (args + result) to the ``tool_calls``
table for the Tools inspector panel.

Design:
  * ``Tool`` wraps a Python callable with a name, description, and JSON-schema
    parameters. ``to_spec()`` emits the OpenAI/Gemma-style tool spec that goes
    into ``engine.generate(tools=...)``.
  * ``ToolRegistry`` holds tools, generates the spec list, dispatches calls, and
    enforces enable/disable + safety (the exec/shell tool is off by default).
  * Reference tools: ``file_search`` (over the RAG memory), ``current_time``,
    and a disabled-by-default ``run_shell`` stub.
"""
from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from store import Store, now_ms


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict            # JSON schema (type: object, properties, required)
    func: Callable[..., Any]
    enabled: bool = True
    dangerous: bool = False     # requires explicit enable (e.g. shell)

    def to_spec(self) -> dict:
        """Emit a function-calling tool spec (OpenAI/Gemma compatible)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    ok: bool
    result: Any = None
    error: Optional[str] = None


class ToolRegistry:
    """Holds tools, builds specs, dispatches + logs calls."""

    def __init__(self, store: Store):
        self.store = store
        self._tools: dict[str, Tool] = {}

    # ---- registration ----------------------------------------------------
    def register(self, tool: Tool) -> None:
        # A user's enable/disable choice must survive a restart (and an app
        # update, which only replaces the .app bundle — this store lives
        # under ~/.aria, untouched by that). Tools are re-registered with
        # their hardcoded defaults on every process start, so restore any
        # persisted override here rather than trusting the caller's default.
        saved = self.store.get_meta(f"tool_enabled:{tool.name}")
        if saved is not None:
            tool.enabled = saved == "1"
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def set_enabled(self, name: str, enabled: bool) -> None:
        if name in self._tools:
            self._tools[name].enabled = enabled
            self.store.set_meta(f"tool_enabled:{name}", "1" if enabled else "0")

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        return [
            {"name": t.name, "description": t.description,
             "enabled": t.enabled, "dangerous": t.dangerous,
             "parameters": t.parameters}
            for t in self._tools.values()
        ]

    def specs(self, enabled_only: bool = True) -> list[dict]:
        """Tool specs to pass into engine.generate(tools=...)."""
        return [t.to_spec() for t in self._tools.values()
                if t.enabled or not enabled_only]

    # ---- dispatch --------------------------------------------------------
    def dispatch(self, name: str, arguments: dict,
                 turn_id: Optional[str] = None) -> ToolResult:
        """Execute a tool call, logging it regardless of outcome."""
        t0 = time.time()
        tool = self._tools.get(name)
        status, result_obj, error = "ok", None, None

        if tool is None:
            status, error = "error", f"unknown tool: {name}"
        elif not tool.enabled:
            status, error = "denied", f"tool disabled: {name}"
        elif tool.dangerous and not tool.enabled:
            status, error = "denied", f"dangerous tool not enabled: {name}"
        else:
            try:
                result_obj = tool.func(**arguments)
            except TypeError as e:
                status, error = "error", f"bad arguments: {e}"
            except Exception as e:
                status, error = "error", str(e)

        duration_ms = int((time.time() - t0) * 1000)
        self._log(name, arguments, result_obj, status, error, duration_ms, turn_id)
        if status == "ok":
            return ToolResult(ok=True, result=result_obj)
        return ToolResult(ok=False, error=error)

    def _log(self, name, arguments, result, status, error, duration_ms, turn_id):
        result_str = None
        if result is not None:
            result_str = json.dumps(result) if not isinstance(result, str) else result
        if error:
            result_str = f"ERROR: {error}"
        self.store.conn.execute(
            "INSERT INTO tool_calls (id,tool_name,arguments,result,status,"
            "duration_ms,turn_id,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), name, json.dumps(arguments),
             (result_str or "")[:4000], status, duration_ms, turn_id, now_ms()),
        )
        self.store.conn.commit()

    def call_log(self, limit: int = 100) -> list[dict]:
        return [dict(r) for r in self.store.query(
            "SELECT id,tool_name,arguments,result,status,duration_ms,created_at "
            "FROM tool_calls ORDER BY created_at DESC LIMIT ?", (limit,))]


# --------------------------------------------------------------------------
# Reference tools
# --------------------------------------------------------------------------
def make_file_search_tool(memory) -> Tool:
    """Search the user's indexed memory (RAG store)."""
    def file_search(query: str, k: int = 5) -> list[dict]:
        hits = memory.retrieve(query, k=k)
        return [{"source": h.source, "text": h.text[:500],
                 "score": round(h.score, 4)} for h in hits]

    return Tool(
        name="file_search",
        description="Search the user's indexed documents and notes for relevant "
                    "passages. Use this to ground answers in the user's own data.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "what to search for"},
                "k": {"type": "integer", "description": "number of results",
                      "default": 5},
            },
            "required": ["query"],
        },
        func=file_search,
    )


def make_current_time_tool() -> Tool:
    def current_time(timezone: str = "local") -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())

    return Tool(
        name="current_time",
        description="Get the current local date and time.",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "default": "local"},
            },
            "required": [],
        },
        func=current_time,
    )


def make_shell_tool() -> Tool:
    """DISABLED by default. A sandboxed command runner — opt-in only."""
    def run_shell(command: str, timeout: int = 10) -> dict:
        proc = subprocess.run(command, shell=True, capture_output=True,
                              text=True, timeout=timeout)
        return {"returncode": proc.returncode,
                "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}

    return Tool(
        name="run_shell",
        description="Run a shell command on the local machine. Use with caution.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "default": 10},
            },
            "required": ["command"],
        },
        func=run_shell,
        enabled=False,      # OFF by default — security
        dangerous=True,
    )


def make_web_search_tool() -> Tool:
    """DISABLED by default — the only tool that sends data off this Mac."""
    from web_search import search_web

    def web_search(query: str, k: int = 5) -> list[dict]:
        return search_web(query, k=k)

    return Tool(
        name="web_search",
        description="Search the public web via DuckDuckGo and return the top "
                    "results (title, snippet, URL). Sends your query to "
                    "DuckDuckGo's servers — the only capability in Aria that "
                    "leaves this Mac. Off by default.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search query"},
                "k": {"type": "integer", "description": "number of results", "default": 5},
            },
            "required": ["query"],
        },
        func=web_search,
        enabled=False,      # OFF by default — privacy
        dangerous=True,     # flagged distinctly in the Tools panel like run_shell
    )


def make_image_generation_tool(images_dir: str) -> Tool:
    """DISABLED by default — not a privacy concern (generation is fully
    local, see image_gen.py), but a real resource one: a ~4.3GB one-time
    model download, then real GPU time and battery per image. Off until the
    user deliberately opts in, same reasoning as web_search being off for a
    different resource (network) rather than a safety one."""
    from image_gen import generate as generate_image

    def image_generation(prompt: str) -> dict:
        return generate_image(prompt, images_dir)

    return Tool(
        name="image_generation",
        description="Generate an image from a text prompt, fully on-device "
                    "via a local diffusion model (FLUX.2 Klein). First use "
                    "downloads a ~4.3GB model. Off by default — real GPU "
                    "time and battery cost per image, and a large one-time "
                    "download.",
        parameters={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "what to generate"},
            },
            "required": ["prompt"],
        },
        func=image_generation,
        enabled=False,      # OFF by default — resource cost, not privacy
        dangerous=False,
    )


def default_registry(store: Store, memory=None, images_dir: str = None) -> ToolRegistry:
    """Build a registry with the standard reference tools."""
    reg = ToolRegistry(store)
    if memory is not None:
        reg.register(make_file_search_tool(memory))
    reg.register(make_current_time_tool())
    reg.register(make_shell_tool())   # present but disabled
    reg.register(make_web_search_tool())  # present but disabled
    if images_dir is not None:
        reg.register(make_image_generation_tool(images_dir))  # present but disabled
    return reg
