from dataclasses import dataclass
from typing import List

from services.policy.policy_config import response_prompt_file_for

CONTROL_MODEL    = "gpt-4.1-mini"
PLAN_MODEL       = "gpt-4o-mini"
GENERATION_MODEL = "gpt-4.1-mini"
CHECK_MODEL      = "gpt-4o-mini"
REWRITE_MODEL    = "gpt-4o-mini"

# Support levels where a safety/leak check is meaningful.
SAFETY_CHECK_LEVELS = {"PARTIAL", "EXPLAIN", "STRUCTURE", "EVALUATION"}

BASE_PROMPT_FILES = {
    "tutor_structure":       "base/srl_tutor_structure.txt",
    "phase_forethought":     "phases/forethought_final.txt",
    "phase_performance":     "phases/performance_final.txt",
    "phase_reflection":      "phases/reflection_final.txt",
    
    "check_reply":           "chains/check_solution_leak_final.txt",
    "rewrite_reply":         "chains/fallback_rewrite_final.txt",
    "checkpoint_and_decide": "chains/student_state_final.txt",
    
    "file_handler":          "constraints/file.txt",
    "missing_file":          "constraints/missing_files.txt",
}


@dataclass
class CheckpointResult:
    """Structured diagnosis of the student's current state from the checkpoint LLM call."""

    request_kind: str
    task_stage: str
    progress_state: str
    has_attempt: bool
    context_gap: str
    expertise_level: str
    frustration_level: str
    srl_focus: str
    subtask_scope: str
    confidence: float
    rationale: List[str]
    parse_ok: bool = True


@dataclass
class SupportDecision:
    """Support strategy selected for the current turn."""

    support_level: str
    response_prompt_file: str
    can_show_code: bool
    must_end_with_question: bool
    should_request_attempt: bool
    confidence: float
    rationale: List[str]
    support_depth: str = "SUBSTANTIVE"
    parse_ok: bool = True


@dataclass
class CheckResult:
    """Outcome of the safety and solution-leak check on a draft reply."""

    is_safe: bool
    leaks_solution: bool
    skipped_diagnosis: bool
    reason: str
    was_skipped: bool = False


def _fallback_checkpoint() -> CheckpointResult:
    """Return a safe default CheckpointResult when the LLM response cannot be parsed."""
    return CheckpointResult(
        request_kind="PRODUCT",
        task_stage="WORKING",
        progress_state="MOVING",
        has_attempt=False,
        context_gap="SMALL",
        expertise_level="NOVICE",
        frustration_level="LOW",
        srl_focus="STRATEGY",
        subtask_scope="NEW_TASK",
        confidence=0.0,
        rationale=["PARSE_FAILED — fallback values in use"],
        parse_ok=False,
    )


def _fallback_decision() -> SupportDecision:
    """Return a safe default SupportDecision when the LLM response cannot be parsed."""
    return SupportDecision(
        support_level="QUESTION",
        response_prompt_file=response_prompt_file_for("QUESTION"),
        can_show_code=False,
        must_end_with_question=True,
        should_request_attempt=False,
        confidence=0.0,
        rationale=["PARSE_FAILED — fallback values in use"],
        support_depth="SUBSTANTIVE",
        parse_ok=False,
    )
