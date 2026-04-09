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
    SupportLevel.CLARIFY:    "responses/old_responses/respond_clarify_v1.txt",
    SupportLevel.QUESTION:   "responses/respond_question_v3.txt",
    SupportLevel.HINT:       "responses/old_responses/respond_hint_v1.txt",       
    SupportLevel.STRUCTURE:  "responses/old_responses/respond_structure_v1.txt",
    SupportLevel.EXPLAIN:    "responses/old_responses/respond_explain_v1.txt",
    SupportLevel.PARTIAL:    "responses/old_responses/respond_partial_v2.txt",
    SupportLevel.REFLECT:    "responses/old_responses/respond_reflect_v1.txt",
    SupportLevel.EVALUATION: "responses/old_responses/respond_evaluation_v1.txt",
}


def response_prompt_file_for(support_level: SupportLevel | str) -> str:
    try:
        level = (
            support_level
            if isinstance(support_level, SupportLevel)
            else SupportLevel(str(support_level).upper())
        )
    except Exception:
        level = SupportLevel.QUESTION
    return RESPONSE_PROMPT_FILES[level]