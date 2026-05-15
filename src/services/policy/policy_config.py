from lib.enums import Phase, SupportLevel

TRANSITIONS = {
    Phase.FORETHOUGHT: {Phase.FORETHOUGHT, Phase.PERFORMANCE, Phase.REFLECTION},
    Phase.PERFORMANCE: {Phase.PERFORMANCE, Phase.REFLECTION},
    Phase.REFLECTION: {Phase.REFLECTION, Phase.PERFORMANCE, Phase.FORETHOUGHT},
}

STAY_THRESHOLD = 0.55
SWITCH_THRESHOLD = 0.70

QUESTION_REQUIRED_LEVELS = {
    SupportLevel.CLARIFY,
    SupportLevel.QUESTION,
    SupportLevel.HINT,
    SupportLevel.STRUCTURE,
    SupportLevel.REFLECT,
}

CODE_ALLOWED_LEVELS = {
    SupportLevel.PARTIAL,
    SupportLevel.EXPLAIN,
}

RESPONSE_PROMPT_FILES = {
    SupportLevel.CLARIFY:    "responses/respond_clarify_final.txt",
    SupportLevel.QUESTION:   "responses/respond_question_final.txt",
    SupportLevel.HINT:       "responses/respond_hint_final.txt",       
    SupportLevel.STRUCTURE:  "responses/respond_structure_final.txt",
    SupportLevel.EXPLAIN:    "responses/respond_explain_final.txt",
    SupportLevel.PARTIAL:    "responses/respond_partial_final.txt",
    SupportLevel.REFLECT:    "responses/respond_reflect_final.txt",
    SupportLevel.EVALUATION: "responses/respond_evaluation_final.txt",
}


def response_prompt_file_for(support_level: SupportLevel | str) -> str:
    """Return the response prompt file path for the given support level."""
    try:
        level = (
            support_level
            if isinstance(support_level, SupportLevel)
            else SupportLevel(str(support_level).upper())
        )
    except Exception:
        level = SupportLevel.QUESTION
    return RESPONSE_PROMPT_FILES[level]