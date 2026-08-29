"""Shared helpers for the local OpenAI-compatible HTTP adapters."""

from __future__ import annotations

import json
from typing import Any


TOOL_CALL_ID_SAFE_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.:"
)


class ClientDisconnected(Exception):
    """The HTTP client closed the socket while a response was being written."""


def normalize_tool_call_id(value: Any) -> str:
    raw = str(value or "").strip()
    normalized = "".join(
        character if character in TOOL_CALL_ID_SAFE_CHARS else "_" for character in raw
    )
    return normalized[:240]


def content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            if part.get("type") in {"text", "input_text", "output_text"}:
                parts.append(str(part.get("text", "")))
            elif part.get("type") in {"image_url", "input_image"}:
                parts.append("[image input omitted by local adapter]")
            else:
                parts.append(json.dumps(part, ensure_ascii=False))
    return "\n".join(parts)
