"""Tool registry + OpenAI-compatible function schema.

A **Tool** is:
    * a JSON-schema description (for LLM tool-calling),
    * a Python function ``fn(env, **args) -> dict`` that executes the action,
    * a side-channel flag ``read_only`` used by the validation layer.

A **Registry** bundles tools into one place and exposes:
    * ``openai_schema()`` — the list you pass to ``LLMClient.chat(tools=...)``
    * ``dispatch(env, name, arguments)`` — run a single LLM tool call and
      always return a dict (success or validated error). The orchestrator
      feeds the dict back as a ``role='tool'`` message.

Everything here is provider-agnostic: the OpenAI SDK's ``tools`` field is
supported identically by Claude, DeepSeek, Qwen, and local vLLM/Ollama.
"""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable


ToolFn = Callable[..., dict[str, Any]]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]      # JSON-schema
    fn: ToolFn
    read_only: bool = False          # observation tools are read-only

    def openai_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class Registry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' already registered.")
        self.tools[tool.name] = tool

    def extend(self, tools: list[Tool]) -> None:
        for t in tools:
            self.add(t)

    def openai_schema(self) -> list[dict[str, Any]]:
        return [t.openai_spec() for t in self.tools.values()]

    def names(self) -> list[str]:
        return list(self.tools.keys())

    def dispatch(self, env: Any, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run a tool by name with validated kwargs. Always returns a dict."""
        tool = self.tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"unknown tool '{name}'",
                    "available": self.names()}
        try:
            result = tool.fn(env, **arguments)
            if not isinstance(result, dict):
                result = {"ok": True, "value": result}
            result.setdefault("ok", True)
            result["_tool"] = name
            return result
        except TypeError as e:
            return {"ok": False, "error": f"bad arguments: {e}", "_tool": name,
                    "expected_schema": tool.parameters}
        except Exception as e:
            return {"ok": False, "error": str(e), "_tool": name,
                    "traceback": traceback.format_exc(limit=3)}


def jsonify(x: Any) -> str:
    """Stable dump used when serializing tool results back to the LLM."""
    return json.dumps(x, default=str, ensure_ascii=False, indent=None)
