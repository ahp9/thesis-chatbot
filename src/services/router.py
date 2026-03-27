import json
import logging
from typing import Any, Dict

from lib.enums import Phase
from services.policy.policy_config import SWITCH_THRESHOLD, TRANSITIONS
from services.prompt_loader import load_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROUTER_MODEL = "gpt-4o-mini"


def _coerce_phase(value: str | None) -> Phase:
    try:
        return Phase((value or "FORETHOUGHT").upper())
    except ValueError:
        return Phase.FORETHOUGHT


def update_phase(
    current_phase: str, predicted_phase: str, confidence: float
) -> str:
    current = _coerce_phase(current_phase)
    predicted = _coerce_phase(predicted_phase or current.value)
    confidence = float(confidence or 0.0)

    if confidence < SWITCH_THRESHOLD:
        return current.value

    if predicted in TRANSITIONS[current]:
        return predicted.value

    return current.value


async def route_message(
    client, user_message: str, llm_history: list, current_phase: str
) -> Dict[str, Any]:
    # Direct instruction prompting is still appropriate for a narrow JSON routing task.
    # [Ch. 3.1.1, p. 38]
    router_system = load_prompt("base/router/router_system_prompt_v6.txt")

    recent = llm_history[-6:] if llm_history else []
    context_lines = []
    for item in recent:
        role = item.get("role", "").upper()
        content = (item.get("content") or "").strip()
        if content:
            context_lines.append(f"{role}: {content}")

    context_text = (
        "\n".join(context_lines) if context_lines else "(no prior context)"
    )

    router_input = f"""RECENT_CONTEXT (most recent messages):
{context_text}

CURRENT_PHASE_STATE: {current_phase}

CURRENT_USER_MESSAGE:
{user_message}
"""

    resp = await client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": router_system},
            {"role": "user", "content": router_input},
        ],
        temperature=0,
    )

    content = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {
            "phase": current_phase or "PERFORMANCE",
            "srl_signal": "NONE",
            "confidence": 0.2,
            "signals": ["router_json_parse_failed"],
            "one_optional_question": "",
        }

    data["phase"] = (data.get("phase") or "PERFORMANCE").upper()
    data["srl_signal"] = (data.get("srl_signal") or "NONE").upper()
    return data
