import logging
from typing import Dict, List, Optional

from services.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

_SUPPORT_LEVEL_ORDER: List[str] = [
    "CLARIFY", "QUESTION", "HINT", "STRUCTURE", "EXPLAIN", "PARTIAL",
    "REFLECT", "EVALUATION",
]

# ---------------------------------------------------------------------------
# Coherence instructions
#
# Also remain in Python because they are computed from a runtime comparison
# between current and previous support levels — not a static value.
# ---------------------------------------------------------------------------

_COHERENCE_INSTRUCTIONS: Dict[str, str] = {
    "same_level_first_repeat": (
        "COHERENCE: The previous turn used the same support level. "
        "Do NOT repeat the same framing, hint, or breakdown. "
        "Find a different angle or entry point — the prior approach did not land."
    ),
    "escalated": (
        "COHERENCE: Support has escalated from the previous turn. "
        "A more concrete approach is warranted — do not hold back to the prior level."
    ),
    "de_escalated": (
        "COHERENCE: Support has de-escalated — the student made progress. "
        "Acknowledge the forward movement. Do not re-explain what was already covered."
    ),
    "first_turn": (
        "COHERENCE: First turn. No prior strategy to account for."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_coherence_key(
    current_support_level: str,
    previous_support_level: Optional[str],
) -> str:
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

def build_generation_context(
    support_depth: str,
    frustration_level: str,
    support_level: str,
    phase: str,
    previous_support_level: Optional[str],
    expertise_level: Optional[str] = None,
) -> str:
    """
    Build the dynamic instruction block injected into the generation system prompt.

    The static contracts (tone by expertise level, phase behaviour, affective
    tone, forward-movement rule) are loaded from the prompt file
    srl_generation_respond.txt and filled with three runtime values:
      - {expertise_level}
      - {phase}
      - {frustration_level}

    The dynamic parts that cannot live in a static file are computed here
    and appended after the filled template:
      - Coherence instruction  (depends on current vs previous support level)
      - Expansion permission   (depends on support_depth)

    Args:
        support_depth:          SURFACE / SUBSTANTIVE / DEEP / etc.
        frustration_level:      LOW / MEDIUM / HIGH
        support_level:          HINT / EXPLAIN / PARTIAL / etc.
        phase:                  FORETHOUGHT / PERFORMANCE / REFLECTION
        previous_support_level: None if first turn
        expertise_level:        NOVICE / INTERMEDIATE / ADVANCED
                                Defaults to INTERMEDIATE if not provided.
    """
    expertise_key = (expertise_level or "INTERMEDIATE").upper()
    phase_key = (phase or "PERFORMANCE").upper()
    affective_key = (frustration_level or "LOW").upper()
    support_key = (support_level or "QUESTION").upper()
    depth_key = (support_depth or "SUBSTANTIVE").upper()

    # Load the template and fill the three static placeholders.
    template = load_prompt("base/srl_generation_respond.txt")
    filled = template.replace("{expertise_level}", expertise_key)
    filled = filled.replace("{phase}", phase_key)
    filled = filled.replace("{frustration_level}", affective_key)
    filled = filled.replace("{support_depth}", depth_key)

    # Append computed dynamic instructions.
    coherence_key = _get_coherence_key(support_key, previous_support_level)
    coherence_instr = _COHERENCE_INSTRUCTIONS.get(
        coherence_key, _COHERENCE_INSTRUCTIONS["first_turn"]
    )

    parts = [filled, coherence_instr]

    return "\n\n".join(parts)