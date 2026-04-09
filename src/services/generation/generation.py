import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_SUPPORT_LEVEL_ORDER: List[str] = [
    "CLARIFY", "QUESTION", "HINT", "STRUCTURE", "EXPLAIN", "PARTIAL",
    "REFLECT", "EVALUATION",
]

_DEPTH_INSTRUCTIONS: Dict[str, str] = {
    "SURFACE": (
        "DEPTH: Keep this response brief and orienting. "
        "One concept, one pointer. Do not go beyond what the student needs "
        "to take their very next step."
    ),
    "SURFACE_PLUS": (
        "DEPTH: Give a brief multi-part orientation. "
        "Name the relevant concept and provide one small grounding step or example. "
        "Do not expand into trade-offs yet — establish the basics first."
    ),
    "SUBSTANTIVE": (
        "DEPTH: Engage with the student's actual decision or problem at a "
        "domain-appropriate level. Name specific options, trade-offs, or failure modes "
        "relevant to their case. Do not define terms they already used correctly. "
        "The response must help them choose, evaluate, or act — not restate what they know."
    ),
    "SUBSTANTIVE_PLUS": (
        "DEPTH: Go beyond naming options — engage with the specific trade-offs, "
        "constraints, or design considerations the student is navigating. "
        "This student is past standard orientation. Address multiple considerations "
        "at once, compare alternatives in their specific context, or explain why the "
        "standard approach may or may not fit. Do not ask setup questions."
    ),
    "DEEP": (
        "DEPTH: Respond at a strategic or technical level. Assume full fluency. "
        "Skip procedural steps entirely. Raise a non-obvious consideration, edge case, "
        "design tension, or failure mode the student has not yet surfaced. "
        "Go to where the standard approach breaks, becomes ambiguous, or requires "
        "architectural judgment."
    ),
}

# ---------------------------------------------------------------------------
# Affective instructions
# ---------------------------------------------------------------------------

_AFFECTIVE_INSTRUCTIONS: Dict[str, str] = {
    "LOW": (
        "TONE: Student appears calm and engaged. "
        "Maintain a direct, focused tone. Prioritise intellectual challenge over reassurance."
    ),
    "MEDIUM": (
        "TONE: Student shows signs of uncertainty or mild frustration. "
        "Acknowledge the difficulty briefly before moving forward. "
        "Keep the next step clear and achievable."
    ),
    "HIGH": (
        "TONE: Student is clearly frustrated or stuck. "
        "Do not add clarifying questions. Do not ask them to try again without guidance. "
        "Provide enough concrete direction for one clear immediate action. "
        "Acknowledge the difficulty briefly, then move quickly to the help."
    ),
}

# ---------------------------------------------------------------------------
# Coherence instructions
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
# Expansion permission
# ---------------------------------------------------------------------------

_EXPANSION_PERMISSION: Dict[str, str] = {
    "SURFACE": "",
    "SURFACE_PLUS": "",
    "SUBSTANTIVE": (
        "FORMAT: A single paragraph is sufficient if it covers the key decision clearly. "
        "A brief setup paragraph followed by the support move is also acceptable. "
        "Do not pad. The student should still have meaningful work left to do."
    ),
    "SUBSTANTIVE_PLUS": (
        "FORMAT: A single paragraph is NOT sufficient. Structure in up to three parts:\n"
        "  1. CONCEPT — the specific idea or decision (2–4 sentences, no generic definitions)\n"
        "  2. ADJACENT EXAMPLE — one small concrete example clearly NOT the student's task\n"
        "     (3–6 lines; code only if can_show_code=true)\n"
        "  3. SUPPORT MOVE — the hint, partial scaffold, or question for this level\n"
        "Keep each part short. The student must still have important reasoning left to do."
    ),
    "DEEP": (
        "FORMAT: Structure in up to three parts:\n"
        "  1. TECHNICAL FRAMING — the specific constraint, trade-off, or failure mode "
        "     (2–3 sentences; assume full domain fluency)\n"
        "  2. ILLUSTRATIVE CASE — a tight example that reveals why the standard approach "
        "     does not simply apply (4–8 lines; code only if can_show_code=true)\n"
        "  3. DIRECTED QUESTION OR PARTIAL — consequence question or partial scaffold "
        "     with key decision left open\n"
        "Do not explain what they already demonstrated they know."
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
) -> str:
    """
    Build the dynamic instruction block injected into the generation system prompt.

    Combines four computed instructions:
      1. Depth      — how much detail and technicality to use
      2. Affective  — how to calibrate tone to the student's emotional state
      3. Coherence  — how to relate this turn to the previous support strategy
      4. Expansion  — whether a multi-part response structure is allowed/required

    Phase tone is now owned by the phase prompt files and is NOT included here.

    Args:
        support_depth:          SURFACE / SUBSTANTIVE / DEEP / etc.
        frustration_level:      LOW / MEDIUM / HIGH
        support_level:          HINT / EXPLAIN / PARTIAL / etc.
        phase:                  FORETHOUGHT / PERFORMANCE / REFLECTION (unused here;
                                retained for interface compatibility)
        previous_support_level: None if first turn
    """
    depth_key = (support_depth or "SUBSTANTIVE").upper()
    affective_key = (frustration_level or "LOW").upper()
    support_key = (support_level or "QUESTION").upper()

    depth_instr = _DEPTH_INSTRUCTIONS.get(depth_key, _DEPTH_INSTRUCTIONS["SUBSTANTIVE"])
    affective_instr = _AFFECTIVE_INSTRUCTIONS.get(affective_key, _AFFECTIVE_INSTRUCTIONS["LOW"])

    coherence_key = _get_coherence_key(support_key, previous_support_level)
    coherence_instr = _COHERENCE_INSTRUCTIONS.get(coherence_key, _COHERENCE_INSTRUCTIONS["first_turn"])

    expansion = _EXPANSION_PERMISSION.get(depth_key, "")

    parts = [depth_instr, affective_instr, coherence_instr]
    if expansion:
        parts.append(expansion)

    return "\n\n".join(parts)