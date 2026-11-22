"""Structured event logging and metrics for the support agents.

Every event is one JSON object per line in solution/logs/agent_events.jsonl:

    {"ts": "...", "level": "info", "event": "kb_retrieval", "thread_id": "t1",
     "ticket_id": "...", "node": "resolve", "top_score": 0.56, "confident": true}

What is logged: routing decisions, classification, knowledge base searches, tool calls and
their outcome, escalations and ticket outcomes. What is never logged: message text, names,
email addresses, free-text reasons or keys. Fields with sensitive names are replaced by their
length, and any email address inside a string value is masked.

    log_event("classified", category="technical", urgency="high")
    summarize_logs()   # counts: knowledge base success vs escalation, tool usage, routing
"""

import json
import logging
import re
from collections import Counter, defaultdict
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# solution/agentic/observability.py -> parents[1] == solution/
LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "agent_events.jsonl"

# Field names whose values may hold personal data or secrets. Only their length is logged.
SENSITIVE_KEYS = frozenset(
    {
        "email", "name", "customer_name", "user_name", "content", "text", "message",
        "messages", "reply", "answer", "issue", "when", "activity", "query", "reason_text",
        "password", "token", "secret", "api_key", "apikey", "authorization", "value",
    }
)  # fmt: skip
EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
MAX_STRING_LENGTH = 200

_context: ContextVar[dict] = ContextVar("agent_log_context", default={})
_logger = logging.getLogger("agentic.events")
_configured_path: Path | None = None


# --------------------------------------------------------------------------- #
# Writing events
# --------------------------------------------------------------------------- #
def configure_logging(path: str | Path | None = None) -> Path:
    """Send events to `path` (default solution/logs/agent_events.jsonl). Safe to call again."""
    global _configured_path
    target = Path(path) if path else LOG_PATH
    if _configured_path == target and _logger.handlers:
        return target
    for handler in list(_logger.handlers):
        _logger.removeHandler(handler)
        handler.close()
    target.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(target, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))  # the message is already JSON
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    _logger.propagate = False  # do not echo the events through the root logger
    _configured_path = target
    return target


def bind_context(**fields: Any):
    """Attach fields (thread_id, ticket_id, node) to every event logged until reset_context."""
    merged = {**_context.get(), **{k: v for k, v in fields.items() if v is not None}}
    return _context.set(merged)


def reset_context(token) -> None:
    _context.reset(token)


def _clean(key: str, value: Any) -> dict[str, Any]:
    """Return the field(s) to write for `key`, with sensitive data removed."""
    if key.lower() in SENSITIVE_KEYS:
        length = len(value) if isinstance(value, (str, list, dict)) else 0
        return {f"{key}_chars": length}
    if isinstance(value, str):
        value = EMAIL_PATTERN.sub("<email>", value)[:MAX_STRING_LENGTH]
    elif isinstance(value, (list, tuple)):
        value = [
            EMAIL_PATTERN.sub("<email>", v)[:MAX_STRING_LENGTH] if isinstance(v, str) else v
            for v in value
        ]
    elif isinstance(value, dict):
        value = {k: v for field, item in value.items() for k, v in _clean(str(field), item).items()}
    return {key: value}


def log_event(event: str, level: str = "info", **fields: Any) -> None:
    """Write one structured event. Never raises: logging must not break a ticket."""
    try:
        if not _logger.handlers:
            configure_logging()
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": level,
            "event": event,
        }
        for key, value in {**_context.get(), **fields}.items():
            record.update(_clean(key, value))
        _logger.info(json.dumps(record, default=str))
    except Exception:  # pragma: no cover - logging must never break the caller
        logging.getLogger(__name__).exception("could not write event %s", event)


def tool_outcome(payload: dict | None) -> str:
    """"ok", "not_found" (an expected miss, such as a first-time customer) or "error"."""
    if payload and payload.get("ok") is True:
        return "ok"
    if payload and payload.get("error") == "not_found":
        return "not_found"
    return "error"


def log_tool_results(agent: str, messages: list, duration_ms: int | None = None) -> int:
    """Log one `tool_call` event per tool result in an agent's messages. Returns how many."""
    from agentic.tool_results import tool_results  # local import: avoids a cycle at load time
    from langchain_core.messages import ToolMessage

    count = 0
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        count += 1
        payload = next(iter(tool_results([message], message.name or "")), None)
        outcome = tool_outcome(payload)
        log_event(
            "tool_call",
            agent=agent,
            tool=message.name,
            outcome=outcome,
            error_code=(payload or {}).get("error") if outcome != "ok" else None,
        )
    return count


# --------------------------------------------------------------------------- #
# Reading events and metrics
# --------------------------------------------------------------------------- #
def read_events(path: str | Path | None = None, thread_id: str | None = None) -> list[dict]:
    """All logged events, oldest first, optionally only one conversation's."""
    target = Path(path) if path else LOG_PATH
    if not target.exists():
        return []
    events = []
    for line in target.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if thread_id is None or event.get("thread_id") == thread_id:
            events.append(event)
    return events


def summarize_logs(path: str | Path | None = None, thread_id: str | None = None) -> dict:
    """Metrics from the log: outcomes, knowledge base success, tool usage, routing, latency."""
    events = read_events(path, thread_id)

    searches = [e for e in events if e["event"] == "kb_retrieval"]
    confident = sum(1 for e in searches if e.get("confident"))
    resolved = [e for e in events if e["event"] == "ticket_resolved"]
    escalated = [e for e in events if e["event"] == "escalated"]
    finished = len(resolved) + len(escalated)

    tools: dict[str, Counter] = defaultdict(Counter)
    tools_by_agent: dict[str, Counter] = defaultdict(Counter)
    for e in events:
        if e["event"] == "tool_call":
            tools[e["tool"]]["calls"] += 1
            tools[e["tool"]]["not_found"] += e["outcome"] == "not_found"
            tools[e["tool"]]["errors"] += e["outcome"] == "error"
            tools_by_agent[e["agent"]][e["tool"]] += 1

    latency: dict[str, list[int]] = defaultdict(list)
    for e in events:
        if e["event"] == "node_completed":
            latency[e["node"]].append(e["duration_ms"])

    return {
        "events": len(events),
        "conversations": len({e["thread_id"] for e in events if e.get("thread_id")}),
        "knowledge_base": {
            "searches": len(searches),
            "confident": confident,
            "not_confident": len(searches) - confident,
            "success_rate": round(confident / len(searches), 2) if searches else None,
        },
        "outcomes": {
            "resolved": len(resolved),
            "escalated": len(escalated),
            "resolved_share": round(len(resolved) / finished, 2) if finished else None,
            "escalation_reasons": dict(Counter(e.get("reason_code") for e in escalated)),
        },
        "classification": dict(
            Counter(e["category"] for e in events if e["event"] == "classified")
        ),
        "routing": dict(
            Counter(e["next"] for e in events if e["event"] == "routing_decision")
        ),
        "tool_usage": {name: dict(counts) for name, counts in sorted(tools.items())},
        "tool_usage_by_agent": {a: dict(c) for a, c in sorted(tools_by_agent.items())},
        "avg_node_ms": {n: round(sum(v) / len(v)) for n, v in sorted(latency.items())},
        "errors": sum(1 for e in events if e.get("level") == "error"),  # logged failures
    }
