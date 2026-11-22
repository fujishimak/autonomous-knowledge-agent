"""Helpers shared by every MCP server: response shapes, input checks and error handling.

Every tool returns a plain dict:
    success: {"ok": True, ...fields}
    failure: {"ok": False, "error": <code>, "message": <short text>}
"""

import functools
import logging
from datetime import date, datetime
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Error codes agents can branch on.
INVALID_INPUT = "invalid_input"
NOT_FOUND = "not_found"
INTERNAL_ERROR = "internal_error"


class ToolInputError(ValueError):
    """Bad tool arguments. The message is written by us, so it is safe to show the LLM."""


def ok(**fields: Any) -> dict:
    return {"ok": True, **fields}


def fail(error: str, message: str) -> dict:
    return {"ok": False, "error": error, "message": message}


def require_text(value: Any, name: str, max_length: int = 200) -> str:
    """Return `value` stripped, or raise ToolInputError if it is blank or too long."""
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        raise ToolInputError(f"{name} must not be empty")
    if len(text) > max_length:
        raise ToolInputError(f"{name} must be at most {max_length} characters")
    return text


def require_int_range(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ToolInputError(f"{name} must be an integer between {low} and {high}")
    return value


def jsonable(value: Any) -> Any:
    """Recursively convert datetimes and enums so a result is plain JSON data."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def pick(source: dict, *keys: str) -> dict:
    """Copy only the listed keys (an allow-list, so new columns are never exposed by accident)."""
    return {key: jsonable(source.get(key)) for key in keys}


def safe_tool(func: Callable[..., dict]) -> Callable[..., dict]:
    """Make a tool never raise.

    ToolInputError becomes an `invalid_input` response. Any other exception is logged
    with its traceback and replaced by a generic message, because exception text
    (SQL, paths, API errors) would otherwise reach the LLM.
    Apply it *under* @server.tool so the tool schema is still built from the signature.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> dict:
        try:
            return func(*args, **kwargs)
        except ToolInputError as exc:
            return fail(INVALID_INPUT, str(exc))
        except Exception:
            logger.exception("tool %s failed", func.__name__)
            return fail(INTERNAL_ERROR, "The request could not be completed. Try again later.")

    return wrapper
