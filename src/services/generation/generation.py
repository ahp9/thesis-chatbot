import logging
from typing import Optional

from utils.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_SUPPORT_LEVEL_ORDER = [
    "CLARIFY", "QUESTION", "HINT", "STRUCTURE", "EXPLAIN", "PARTIAL",
    "REFLECT", "EVALUATION",
]

# ---------------------------------------------------------------------------
# Coherence instructions
#
# Computed at runtime by comparing current vs previous support level.
# Passed into the planner payload so the planner can factor it into
# move1_aim and handback_content decisions.
# ---------------------------------------------------------------------------

_COHERENCE_INSTRUCTIONS = {
    "same_level_first_repeat": (
        "Same support level as last turn. "
        "Do NOT plan the same framing, angle, or breakdown. "
        "Find a different entry point — the prior approach did not land."
    ),
    "escalated": (
        "Support has escalated from the previous turn. "
        "Plan a more concrete move — do not hold back to the prior level."
    ),
    "de_escalated": (
        "Support has de-escalated — the student made progress. "
        "Acknowledge the forward movement. "
        "Do not re-plan anything that was already covered."
    ),
    "first_turn": (
        "First turn. No prior strategy to account for."
    ),
}


def _get_coherence_key(
    current_support_level: str,
    previous_support_level: Optional[str],
) -> str:
    """Map current and previous support levels to a coherence instruction key."""
    if not previous_support_level:
        return "first_turn"
    if current_support_level == previous_support_level:
        return "same_level_first_repeat"
    try:
        curr_idx = _SUPPORT_LEVEL_ORDER.index(current_support_level)
        prev_idx = _SUPPORT_LEVEL_ORDER.index(previous_support_level)
        return "escalated" if curr_idx > prev_idx else "de_escalated"
    except ValueError:
        return "first_turn"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_coherence_instruction(
    current_support_level: str,
    previous_support_level: Optional[str],
) -> str:
    """Return an instruction string flagging whether support has stayed, escalated, or de-escalated relative to the previous turn."""
    key = _get_coherence_key(
        (current_support_level or "QUESTION").upper(),
        (previous_support_level or "").upper() or None,
    )
    return _COHERENCE_INSTRUCTIONS.get(key, _COHERENCE_INSTRUCTIONS["first_turn"])


def build_filled_structure(
    expertise_level: str,
    phase: str,
    srl_focus: str,
    frustration_level: str,
    support_depth: str,
) -> str:
    """Load srl_tutor_structure.txt and substitute the five runtime placeholders for the planner system prompt."""
    template = load_prompt("base/srl_tutor_structure.txt")
    filled = template.replace("{expertise_level}", (expertise_level or "INTERMEDIATE").upper())
    filled = filled.replace("{phase}",             (phase or "PERFORMANCE").upper())
    filled = filled.replace("{srl_focus}",         (srl_focus or "GOAL").upper())
    filled = filled.replace("{frustration_level}", (frustration_level or "LOW").upper())
    filled = filled.replace("{support_depth}",     (support_depth or "SUBSTANTIVE").upper())
    return filled