# -*- coding: utf-8 -*-
"""Secret redaction and bounded diagnostic payload sanitization."""

from dataclasses import dataclass
import re
from typing import Any, Optional


SECRET_PATTERNS = [
    re.compile(
        r"(?i)(api[_-]?key|secret|token|password|bearer|authorization|auth)[\s:=]+['\"]?([a-zA-Z0-9_\-.]{8,})['\"]?"
    ),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"(?i)bearer\s+[a-zA-Z0-9_\-.]{20,}"),
]


@dataclass(frozen=True)
class RedactionPolicy:
    redact_secrets: bool = True
    include_chunk_content: bool = False
    include_full_context: bool = False
    include_prompts: bool = False
    max_preview_chars: int = 200


def sanitize_text(text: Optional[str], policy: Optional[RedactionPolicy] = None) -> str:
    if text is None:
        return ""
    value = text if isinstance(text, str) else str(text)
    policy = policy or RedactionPolicy()

    if policy.redact_secrets:
        for pattern in SECRET_PATTERNS:
            value = pattern.sub("[REDACTED_SECRET]", value)

    if not policy.include_chunk_content and len(value) > policy.max_preview_chars:
        value = value[: policy.max_preview_chars] + "... [TRUNCATED]"
    return value


def sanitize_dict(data: Any, policy: Optional[RedactionPolicy] = None) -> Any:
    """Recursively sanitize keys and string values, including metadata payloads."""
    policy = policy or RedactionPolicy()

    def is_secret_key(key_lower: str) -> bool:
        if key_lower in {"key", "secret", "token", "password", "authorization", "auth"}:
            return True
        return bool(re.search(
            r"(?:^|_)(?:api|access|private)?_?(?:key|secret|token|password|authorization|auth)(?:_|$)",
            key_lower,
        ))

    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            key_text = str(key)
            key_lower = key_text.lower()
            if policy.redact_secrets and is_secret_key(key_lower):
                cleaned[key_text] = "[REDACTED_SECRET]"
            else:
                cleaned[key_text] = sanitize_dict(value, policy)
        return cleaned
    if isinstance(data, list):
        return [sanitize_dict(item, policy) for item in data]
    if isinstance(data, tuple):
        return [sanitize_dict(item, policy) for item in data]
    if isinstance(data, str):
        return sanitize_text(data, policy)
    return data
