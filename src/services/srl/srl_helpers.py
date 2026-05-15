import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from utils.history_adapter import build_learning_trajectory, recent_control_state
from services.srl.srl_models import (
    BASE_PROMPT_FILES,
    CONTROL_MODEL,
    SAFETY_CHECK_LEVELS,
    CheckpointResult,
    SupportDecision,
)

logger = logging.getLogger(__name__)


def extract_json(raw: str) -> Dict[str, Any]:
    """Parse JSON from a raw model response, trying direct parse then fence and bare-brace extraction."""
    if not raw or not raw.strip():
        logger.warning("_extract_json: empty response from model")
        return {}

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    logger.error("_extract_json: could not parse model response. First 500 chars:\n%s", raw[:500])
    return {}


async def call_json(
    client,
    system_prompt: str,
    user_prompt: str,
    model: str = CONTROL_MODEL,
) -> Tuple[str, Dict[str, Any], bool]:
    """Call the LLM in JSON mode and return (raw_text, parsed_dict, parse_ok)."""
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    data = extract_json(raw)
    return raw, data, bool(data)


def phase_prompt_file(phase: Optional[str]) -> str:
    """Return the prompt file path for the given SRL phase string."""
    phase = (phase or "PERFORMANCE").upper()
    if phase == "FORETHOUGHT":
        return BASE_PROMPT_FILES["phase_forethought"]
    if phase == "REFLECTION":
        return BASE_PROMPT_FILES["phase_reflection"]
    return BASE_PROMPT_FILES["phase_performance"]


def has_file_content(user_message: str) -> bool:
    """Return True if the message contains an embedded file block."""
    markers = ["FILE:", "FILE_BLOCK:", "FILES:", "CURRENT_USER_INPUT_WITH_FILES:"]
    return any(marker in user_message for marker in markers)


def compact_history(llm_history: List[Dict[str, Any]], limit: int = 8) -> str:
    """Render the most recent `limit` turns as a compact ROLE: content string."""
    recent = llm_history[-limit:] if llm_history else []
    lines = []
    for m in recent:
        role    = (m.get("role") or "user").upper()
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content[:500]}")
    return "\n".join(lines) if lines else "(no prior context)"


def build_native_history(
    llm_history: List[Dict[str, Any]],
    limit: int = 8,
) -> List[Dict[str, str]]:
    """Build the message list for the LLM, replacing out-of-window file uploads with placeholders."""
    # Track the most recent file upload so its content is preserved
    last_file_idx = None
    for i, m in enumerate(llm_history):
        if m.get("role") == "user" and "--- FILE:" in (m.get("content") or ""):
            last_file_idx = i

    window_start = max(0, len(llm_history) - (limit * 2))

    clean: List[Dict[str, str]] = []
    for i, m in enumerate(llm_history[-(limit * 2):]):
        absolute_idx = window_start + i
        role = m.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue

        if "--- FILE:" in content and absolute_idx != last_file_idx:
            # Older upload in the same window — swap full content for a placeholder
            file_names = re.findall(r"--- FILE:\s*(\S+)", content)
            labeled_files = ", ".join(file_names) if file_names else "files"
            text_before = content.split("--- FILE:")[0].strip()
            placeholder = f"[previously uploaded {labeled_files}]"
            content = (text_before + "\n" + placeholder).strip() if text_before else placeholder

        clean.append({"role": role, "content": content})

    clean = clean[-limit:]

    # Ensure the list starts with a user turn
    while clean and clean[0]["role"] == "assistant":
        clean.pop(0)

    return clean


def build_checkpoint_payload(
    route: Dict[str, Any],
    llm_history: List[Dict[str, Any]],
    user_message: str,
) -> str:
    """Assemble the user-turn payload sent to the checkpoint LLM."""
    recent_control = recent_control_state(llm_history)
    trajectory     = build_learning_trajectory(llm_history, limit=3)

    parts = [
        f"CURRENT_PHASE:\n{route.get('phase', 'UNKNOWN')}",
        f"LEARNING_TRAJECTORY (last 3 turns):\n{trajectory}",
        f"PREVIOUS_CONTROL:\n{json.dumps(recent_control, indent=2)}",
        f"RECENT_HISTORY:\n{compact_history(llm_history)}",
        f"CURRENT_USER_MESSAGE:\n{user_message}",
    ]
    return "\n\n".join(parts)


def should_run_safety_check(decision: SupportDecision) -> bool:
    """Return True if the draft reply warrants a safety and solution-leak check."""
    return decision.can_show_code or decision.support_level in SAFETY_CHECK_LEVELS


def strip_plan_block(text: str) -> str:
    """Remove any <plan>...</plan> block left in the writer's output."""
    return re.sub(r"<plan>.*?</plan>", "", text, flags=re.DOTALL).strip()


def build_writer_brief(
    route: Dict[str, Any],
    checkpoint: CheckpointResult,
    decision: SupportDecision,
) -> dict:
    """Build the JSON brief passed to the reply writer describing the current turn constraints."""
    return {
        "phase":                  route.get("phase", "PERFORMANCE"),
        "support_level":          decision.support_level,
        "support_depth":          decision.support_depth,
        "can_show_code":          decision.can_show_code,
        "must_end_with_question": decision.must_end_with_question,
        "request_kind":           checkpoint.request_kind,
        "task_stage":             checkpoint.task_stage,
        "progress_state":         checkpoint.progress_state,
        "context_gap":            checkpoint.context_gap,
        "subtask_scope":          checkpoint.subtask_scope,
        "has_attempt":            checkpoint.has_attempt,
    }


def file_referenced_but_missing(
    user_message: str,
    llm_history: list,
    history_limit: int = 8,
) -> str | None:
    """Return a filename if the user references a file that has scrolled out of the active window, else None."""
    file_keywords = ["file", "csv", "dataset", "data", "upload", "notebook", "script", "code"]
    msg_lower = user_message.lower()

    if has_file_content(user_message):
        return None  # File is present in this message

    if not any(kw in msg_lower for kw in file_keywords):
        return None  # No file reference detected

    # Check if any file upload is still within the active window
    recent = (llm_history or [])[-(history_limit * 2):]
    for item in reversed(recent):
        if "--- FILE:" in (item.get("content") or ""):
            return None

    # File was uploaded but has scrolled out — return its name
    for item in reversed(llm_history or []):
        if "--- FILE:" in (item.get("content") or ""):
            names = re.findall(r"--- FILE: (.+?) \(", item.get("content", ""))
            return names[0] if names else "your uploaded file"

    return None
