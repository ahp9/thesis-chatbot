from __future__ import annotations

from typing import Any, Iterable

from lib.enums import SupportLevel

import re

_CONTINUITY_SIGNALS = [
    r"\bbased on (that|this|the above|my (previous|last|prior))\b",
    r"\bfor the same\b",
    r"\bcontinue\b",
    r"\busing (that|the same|my) (plot|chart|code|output|result|variable)\b",
    r"\bfrom (before|the previous|last time)\b",
    r"\bnow (add|include|update|change)\b",
    r"\bkeep (the same|that)\b",
]

def classify_subtask_scope(user_message: str) -> str:
    """
    Returns 'CONTINUE' if the message explicitly refers to a prior turn,
    'NEW_TASK' otherwise.
    """
    msg = user_message.lower()
    for pattern in _CONTINUITY_SIGNALS:
        if re.search(pattern, msg):
            return "CONTINUE"
    return "NEW_TASK"


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


def build_learning_trajectory(
    llm_history: list[dict[str, Any]],
    limit: int = 3,
) -> str:
    entries: list[str] = []
    turn_num = 0

    for item in reversed(llm_history or []):
        if item.get("role") != "assistant":
            continue
        if turn_num >= limit:
            break

        route = item.get("route") or {}
        checkpoint = item.get("checkpoint") or item.get("diagnosis") or {}
        decision = item.get("decision") or {}

        phase = route.get("phase", "UNKNOWN")
        support_level = decision.get("support_level", "UNKNOWN")
        progress_state = checkpoint.get("progress_state", "UNKNOWN")
        frustration_level = checkpoint.get("frustration_level", "UNKNOWN")
        expertise_level = checkpoint.get("expertise_level", "UNKNOWN")

        entries.append(
            f"phase={phase} | support_level={support_level} | "
            f"progress_state={progress_state} | frustration_level={frustration_level} | "
            f"expertise_level={expertise_level}"
        )
        turn_num += 1

    if not entries:
        return "(no prior turns — treat as turn 1)"

    # Reverse so most recent is last (chronological reading order)
    entries.reverse()
    lines = [f"  turn {i + 1}: {e}" for i, e in enumerate(entries)]
    return "\n".join(lines)