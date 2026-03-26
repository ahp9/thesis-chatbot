import json
import logging
from typing import Any, Dict

from services.prompt_loader import load_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROUTER_MODEL = "gpt-4o-mini"

VALID_PHASES = {"FORETHOUGHT", "PERFORMANCE", "REFLECTION"}

ALLOWED_TRANSITIONS = {
    "FORETHOUGHT": {"FORETHOUGHT", "PERFORMANCE", "REFLECTION"},
    "PERFORMANCE": {"FORETHOUGHT", "PERFORMANCE", "REFLECTION"},
    "REFLECTION": {"FORETHOUGHT", "PERFORMANCE", "REFLECTION"},
}


def update_phase(
    current_phase: str, predicted_phase: str, confidence: float
) -> str:
    current_phase = (current_phase or "FORETHOUGHT").upper()
    predicted_phase = (predicted_phase or current_phase).upper()
    confidence = float(confidence or 0.0)

    if current_phase not in VALID_PHASES:
        current_phase = "FORETHOUGHT"
    if predicted_phase not in VALID_PHASES:
        predicted_phase = current_phase

    # Stay put if confidence is too low
    if confidence < 0.60:
        return current_phase

    # Allow transition only if it is valid from the current phase
    if predicted_phase in ALLOWED_TRANSITIONS[current_phase]:
        return predicted_phase

    return current_phase


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
