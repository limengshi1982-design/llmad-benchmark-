"""Unified LLM client — OpenAI format AND Anthropic format.

Callers always speak OpenAI-shaped messages and OpenAI-shaped tool schemas.
Internally we branch on the profile's ``format`` field:

    format == "openai"     -> openai SDK (default — OpenAI, DeepSeek, Qwen,
                              local vLLM/Ollama, Anthropic's OpenAI-compat
                              endpoint, all go here)
    format == "anthropic"  -> anthropic SDK (Anthropic's native /v1/messages
                              and bridges like MiniMax /anthropic)

The ``LLMResponse`` returned is identical in both paths: it carries
``content`` (assistant text) and ``tool_calls`` in OpenAI shape
``[{'id', 'name', 'arguments'}]``. This lets the tool registry & the
experiment drivers stay protocol-agnostic.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "config" / "models.json"


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = None
    profile: str = ""
    model: str = ""
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stop_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """One client, many providers."""

    def __init__(
        self,
        profile: str | None = None,
        config_path: str | Path = _DEFAULT_CONFIG,
    ) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.profile_name = profile or self.config["default_profile"]
        if self.profile_name not in self.config["profiles"]:
            raise ValueError(
                f"Unknown profile '{self.profile_name}'. "
                f"Available: {', '.join(sorted(self.config['profiles']))}"
            )

        self.profile = self.config["profiles"][self.profile_name]
        self.gen_cfg = self.config.get("generation", {})
        self.format = self.profile.get("format", "openai")
        self.model = self.profile["model"]

        api_key = os.environ.get(self.profile["api_key_env"]) \
            or self.profile.get("api_key_default")
        if not api_key:
            raise RuntimeError(
                f"API key missing. Export env var '{self.profile['api_key_env']}' "
                f"or add 'api_key_default' to profile '{self.profile_name}'."
            )

        if self.format == "openai":
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.profile["base_url"],
                api_key=api_key,
                timeout=self.gen_cfg.get("timeout_s", 120),
                max_retries=self.gen_cfg.get("max_retries", 3),
            )
        elif self.format == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(
                base_url=self.profile["base_url"],
                api_key=api_key,
                timeout=self.gen_cfg.get("timeout_s", 120),
                max_retries=self.gen_cfg.get("max_retries", 3),
            )
        else:
            raise ValueError(
                f"Unsupported format '{self.format}' in profile '{self.profile_name}'."
            )

    # ----------------------------------------------------------- public API
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if self.format == "openai":
            return self._chat_openai(messages, tools, temperature, max_tokens, **kwargs)
        return self._chat_anthropic(messages, tools, temperature, max_tokens, **kwargs)

    def __repr__(self) -> str:
        return f"LLMClient(profile={self.profile_name!r}, model={self.model!r}, format={self.format!r})"

    # ------------------------------------------------------------- openai
    def _chat_openai(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None
            else self.gen_cfg.get("temperature", 0.2),
            "max_tokens": max_tokens if max_tokens is not None
            else self.gen_cfg.get("max_tokens", 2048),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = kwargs.pop("tool_choice", "auto")
        payload.update(kwargs)

        t0 = time.perf_counter()
        raw = self.client.chat.completions.create(**payload)
        latency = time.perf_counter() - t0

        msg = raw.choices[0].message
        tool_calls: list[dict[str, Any]] = []
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(
                    {"id": tc.id, "name": tc.function.name, "arguments": args}
                )

        usage = getattr(raw, "usage", None)
        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            raw=raw,
            profile=self.profile_name,
            model=self.model,
            latency_s=latency,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            stop_reason=getattr(raw.choices[0], "finish_reason", None),
        )

    # ---------------------------------------------------------- anthropic
    def _chat_anthropic(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
        **kwargs: Any,
    ) -> LLMResponse:
        system_text, anth_messages = _to_anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": anth_messages,
            "max_tokens": max_tokens if max_tokens is not None
            else self.gen_cfg.get("max_tokens", 2048),
        }
        if system_text:
            payload["system"] = system_text
        if temperature is not None or "temperature" in self.gen_cfg:
            payload["temperature"] = (
                temperature if temperature is not None else self.gen_cfg["temperature"]
            )
        if tools:
            payload["tools"] = [_openai_tool_to_anthropic(t) for t in tools]
            tool_choice = kwargs.pop("tool_choice", "auto")
            if tool_choice == "auto":
                payload["tool_choice"] = {"type": "auto"}
            elif tool_choice == "none":
                # Anthropic doesn't have a direct 'none', but we can omit tools.
                payload.pop("tools", None)
            elif isinstance(tool_choice, dict):
                payload["tool_choice"] = tool_choice
        payload.update(kwargs)

        t0 = time.perf_counter()
        raw = self.client.messages.create(**payload)
        latency = time.perf_counter() - t0

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in raw.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "arguments": dict(block.input) if block.input else {},
                    }
                )

        usage = getattr(raw, "usage", None)
        return LLMResponse(
            content="".join(text_parts) if text_parts else None,
            tool_calls=tool_calls,
            raw=raw,
            profile=self.profile_name,
            model=self.model,
            latency_s=latency,
            prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            stop_reason=getattr(raw, "stop_reason", None),
        )


# ---------------------------------------------------------------- adapters
def _openai_tool_to_anthropic(spec: dict[str, Any]) -> dict[str, Any]:
    """OpenAI tool schema -> Anthropic tool schema.

    OpenAI: {"type": "function", "function": {"name", "description", "parameters"}}
    Anthropic: {"name", "description", "input_schema"}
    """
    fn = spec.get("function", spec)
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters", {"type": "object"}),
    }


def _to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Translate OpenAI-shaped messages into (system_text, anthropic_messages).

    Mapping rules:
        * 'system' messages are concatenated into a single top-level system string
          (Anthropic takes 'system' outside the messages array).
        * 'user' messages become {'role': 'user', 'content': str|blocks}.
        * 'assistant' messages become {'role': 'assistant', 'content': blocks}.
          Any OpenAI tool_calls are converted to 'tool_use' content blocks.
        * 'tool' messages become USER messages with a 'tool_result' block
          (Anthropic's convention — tool outputs are sent back as user turn).
    """
    system_chunks: list[str] = []
    out: list[dict[str, Any]] = []

    for m in messages:
        role = m.get("role")
        if role == "system":
            content = m.get("content") or ""
            if content:
                system_chunks.append(content)
            continue

        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id"),
                "content": m.get("content") or "",
            }
            # Merge consecutive tool_results into one user turn if possible.
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
            continue

        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            text = m.get("content")
            if text:
                blocks.append({"type": "text", "text": text})
            for tc in m.get("tool_calls", []) or []:
                fn = tc.get("function", tc)
                name = fn.get("name")
                raw_args = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    args = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id"),
                        "name": name,
                        "input": args,
                    }
                )
            if not blocks:
                blocks = [{"type": "text", "text": ""}]
            out.append({"role": "assistant", "content": blocks})
            continue

        # Default: user message (or unknown role treated as user).
        content = m.get("content")
        if isinstance(content, list):
            out.append({"role": "user", "content": content})
        else:
            out.append({"role": "user", "content": content or ""})

    return "\n\n".join(system_chunks), out


def list_profiles(config_path: str | Path = _DEFAULT_CONFIG) -> list[str]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return sorted(cfg["profiles"].keys())
