from __future__ import annotations

from typing import Any, Iterable

from lib.enums import SupportLevel


def iter_assistant_turns(llm_history: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for item in llm_history or []:
        if item.get("role") == "assistant":
            yield item


def recent_support_levels(
    llm_history: list[dict[str, Any]],
    limit: int = 3,
) -> list[SupportLevel]:
    levels: list[SupportLevel] = []
    for item in iter_assistant_turns(llm_history):
        decision = item.get("decision")
        if not isinstance(decision, dict):
            continue

        raw_level = decision.get("support_level")
        try:
            levels.append(SupportLevel(raw_level))
        except Exception:
            continue

    return levels[-limit:]


def recent_control_state(llm_history: list[dict[str, Any]]) -> dict[str, Any]:
    for item in reversed(llm_history or []):
        if item.get("role") != "assistant":
            continue

        checkpoint = item.get("checkpoint") or item.get("diagnosis")
        decision = item.get("decision")

        if not isinstance(checkpoint, dict) or not isinstance(decision, dict):
            continue

        return {
            "previous_progress_state": checkpoint.get("progress_state"),
            "previous_frustration_level": checkpoint.get("frustration_level"),
            "previous_support_level": decision.get("support_level"),
            "previous_support_depth": decision.get("support_depth"),
        }

    return {}


def last_assistant_reply(
    llm_history: list[dict[str, Any]],
    default: str = "(none)",
    max_chars: int = 800,
) -> str:
    for item in reversed(llm_history or []):
        if item.get("role") == "assistant":
            content = (item.get("content") or "").strip()
            return content[:max_chars] if content else default
    return default
