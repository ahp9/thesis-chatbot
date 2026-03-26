from lib.enums import Phase, SupportLevel

TRANSITIONS = {
    Phase.FORETHOUGHT: {Phase.FORETHOUGHT, Phase.PERFORMANCE, Phase.REFLECTION},
    Phase.PERFORMANCE: {Phase.PERFORMANCE, Phase.REFLECTION},
    Phase.REFLECTION: {Phase.REFLECTION, Phase.PERFORMANCE, Phase.FORETHOUGHT},
}

STAY_THRESHOLD = 0.55
SWITCH_THRESHOLD = 0.60

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
    SupportLevel.CLARIFY: "responses/respond_diagnose_v1.txt",
    SupportLevel.QUESTION: "responses/respond_question_v2.txt",
    SupportLevel.HINT: "responses/respond_hint_v1.txt",
    SupportLevel.STRUCTURE: "responses/respond_structure_v1.txt",
    SupportLevel.EXPLAIN: "responses/respond_explain_v1.txt",
    SupportLevel.PARTIAL: "responses/respond_partial_v2.txt",
    SupportLevel.REFLECT: "responses/respond_reflect_v1.txt",
    SupportLevel.EVALUATION: "responses/respond_evaluation_v1.txt",
}