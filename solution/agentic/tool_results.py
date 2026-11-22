"""Read what tools returned out of an agent's message list.

Graph nodes take ids and facts from tool output, never from the agent's prose.
"""

import json
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage


def _parse(content: Any) -> dict | None:
    """MCP tool results arrive as a JSON string or as a list of text blocks holding JSON."""
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def tool_results(messages: list[BaseMessage], tool_name: str) -> list[dict]:
    """Every parsed result of `tool_name` in the messages, oldest first."""
    results = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == tool_name:
            payload = _parse(message.content)
            if payload is not None:
                results.append(payload)
    return results


def ok_results(messages: list[BaseMessage], tool_name: str) -> list[dict]:
    """Only the successful ({"ok": true}) results of `tool_name`."""
    return [r for r in tool_results(messages, tool_name) if r.get("ok") is True]
