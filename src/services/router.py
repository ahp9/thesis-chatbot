import json
import re
import logging
from typing import Any, Dict
from services.prompt_loader import load_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROUTER_MODEL = "gpt-4o-mini"

PHASE_ORDER = ["FORETHOUGHT", "PERFORMANCE", "REFLECTION"]

def update_phase(current_phase: str, predicted_phase: str, confidence: float) -> str:
    """
    SRL phases are states. We keep a current phase and move forward when the router is confident.

    Rules:
    - If confidence is low, stay in current phase.
    - Allow forward movement (forethought->performance->reflection).
    - Avoid bouncing backwards.
    """
    current_phase = (current_phase or "FORETHOUGHT").upper()
    predicted_phase = (predicted_phase or current_phase).upper()
    confidence = float(confidence or 0.0)
    
    logger.info(f"Router predicted phase {predicted_phase} with confidence {confidence:.2f} (current phase: {current_phase})")
    

    if current_phase not in PHASE_ORDER:
        current_phase = "FORETHOUGHT"
    if predicted_phase not in PHASE_ORDER:
        predicted_phase = current_phase

    # If router is unsure, don't change state
    if confidence < 0.60:
        logger.info(f"Router confidence {confidence:.2f} below threshold, staying in {current_phase}")
        return current_phase

    # Move forward only
    if PHASE_ORDER.index(predicted_phase) > PHASE_ORDER.index(current_phase):
        logger.info(f"Router confidence {confidence:.2f} above threshold, moving to {predicted_phase}")
        return predicted_phase

    # Allow reflection if predicted strongly (even if not "forward" due to weird inputs)
    if predicted_phase == "REFLECTION":
        return "REFLECTION"

    return current_phase

async def route_message(client, user_message: str, llm_history: list, current_phase: str) -> Dict[str, Any]:
    router_system = load_prompt("base/router/router_system_prompt_v3.txt")

    # Take last N messages as "immediate context"
    N = 6
    recent = llm_history[-N:] if llm_history else []

    # Build a clean context transcript (avoid dumping huge text)
    context_lines = []
    for m in recent:
        role = m.get("role", "").upper()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        # keep it compact
        context_lines.append(f"{role}: {content}")

    context_text = "\n".join(context_lines) if context_lines else "(no prior context)"

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

    content = resp.choices[0].message.content or ""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {
            "phase": "PERFORMANCE",
            "strategy": "NONE",
            "confidence": 0.2,
            "signals": ["router_json_parse_failed"],
            "one_optional_question": ""
        }

    data["phase"] = (data.get("phase") or "PERFORMANCE").upper()
    data["strategy"] = (data.get("strategy") or "NONE").upper()
    return data