import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from services.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

CHAIN_MODEL = "gpt-4o-mini"


@dataclass
class DiagnosisResult:
    request_kind: str
    student_stage: str
    student_state: str
    has_attempt: bool
    needs_diagnosis: bool
    tool_context_known: bool
    expertise_level: str
    frustration_level: str
    implementation_allowed: bool
    confidence: float
    rationale: List[str]


@dataclass
class SupportDecision:
    support_level: str
    response_prompt_file: str
    can_show_code: bool
    must_end_with_question: bool
    should_request_attempt: bool
    confidence: float
    rationale: List[str]


@dataclass
class CheckResult:
    is_safe: bool
    leaks_solution: bool
    skipped_diagnosis: bool
    reason: str


# Prompt chaining breaks a complex task into smaller sequential prompts whose outputs
# become inputs to later prompts. [Ch. 3.3.1, pp. 68-69]
BASE_PROMPT_FILES = {
    "identity": "base/srl_model_v2.txt",
    "phase_forethought": "phases/forethought_core.txt",
    "phase_performance": "phases/performance_core.txt",
    "phase_reflection": "phases/reflection_core.txt",
    "diagnose": "chains/diagnose_student.txt",
    "decide_support": "chains/choose_support_level.txt",
    "check_reply": "chains/check_solution_leak_v1.txt",
    "rewrite_reply": "chains/fallback_rewrite_v1.txt",
    "diagnose_and_decide": "chains/student_state_v1.txt",
    "file_handler": "base/file.txt",
}

RESPONSE_PROMPT_FILES = {
    "DIAGNOSE": "responses/respond_diagnose_v1.txt",
    "QUESTION": "responses/respond_question_v2.txt",
    "HINT": "responses/respond_hint_v1.txt",
    "STRUCTURE": "responses/respond_structure_v1.txt",
    "EXPLAIN": "responses/respond_explain_v1.txt",
    "PARTIAL": "responses/respond_partial_v1.txt",
    "REFLECT": "responses/respond_reflect_v1.txt",
    "EVALUATION": "responses/respond_evaluation_v1.txt",
}


async def _call_json(client, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    resp = await client.chat.completions.create(
        model=CHAIN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    content = resp.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}

def _phase_prompt_file(phase: str | None) -> str:
    phase = (phase or "PERFORMANCE").upper()
    if phase == "FORETHOUGHT": 
        return BASE_PROMPT_FILES["phase_forethought"]
    if phase == "REFLECTION": 
        return BASE_PROMPT_FILES["phase_reflection"]
    return BASE_PROMPT_FILES["phase_performance"]

def _compact_history(llm_history: List[Dict[str, Any]], limit: int = 8) -> str:
    recent = llm_history[-limit:] if llm_history else []
    lines = [f"{(m.get('role') or 'user').upper()}: {(m.get('content') or '').strip()}" for m in recent if m.get('content')]
    return "\n".join(lines) if lines else "(no prior context)"

def _make_chain_context(
    route: Dict[str, Any], llm_history: List[Dict[str, Any]], user_message: str
) -> str:
    return (
        f"CURRENT_PHASE: {route.get('phase', 'PERFORMANCE')}\n"
        f"ROUTER_STRATEGY: {route.get('strategy', 'NONE')}\n"
        f"ROUTER_CONFIDENCE: {route.get('confidence', 0.0)}\n\n"
        f"RECENT_HISTORY:\n{_compact_history(llm_history)}\n\n"
        f"CURRENT_USER_MESSAGE:\n{user_message}"
    )

def _extract_file_blocks(user_message: str) -> str:
    """
    Keeps this tiny:
    - If upstream already injected FILE: / FILE_BLOCK / CURRENT_USER_INPUT_WITH_FILES,
      preserve it as-is.
    - Otherwise, return empty string.
    """
    markers = [
        "FILE:",
        "FILE_BLOCK:",
        "FILES:",
        "CURRENT_USER_INPUT_WITH_FILES:",
    ]
    if any(marker in user_message for marker in markers):
        return user_message
    return ""

def _build_diagnosis_payload(
    route: Dict[str, Any],
    llm_history: List[Dict[str, Any]],
    user_message: str,
) -> str:
    """
    Match the prompt's expected schema exactly:
    1. CURRENT_PHASE
    2. RECENT_HISTORY
    3. CURRENT_USER_MESSAGE

    If file content already exists in user_message, expose it clearly under FILE_CONTEXT.
    """
    file_context = _extract_file_blocks(user_message)

    parts = [
        f"CURRENT_PHASE:\n{route.get('phase', 'UNKNOWN')}",
        f"RECENT_HISTORY:\n{_compact_history(llm_history)}",
        f"CURRENT_USER_MESSAGE:\n{user_message}",
    ]

    if file_context:
        parts.append(f"FILE_CONTEXT:\n{file_context}")

    return "\n\n".join(parts)


async def diagnose_and_decide(client, route, llm_history, user_message) -> tuple[DiagnosisResult, SupportDecision]:
    # Cache-friendly system prompt: Large static blocks first
    system_prompt = "\n\n".join([
        load_prompt(BASE_PROMPT_FILES["identity"]),
        load_prompt(_phase_prompt_file(route.get("phase"))),
        load_prompt(BASE_PROMPT_FILES["diagnose_and_decide"]),
        load_prompt(BASE_PROMPT_FILES["file_handler"]),
    ])

    payload = _build_diagnosis_payload(route, llm_history, user_message)
    data = await _call_json(client, system_prompt, payload)

    diag_raw = data.get("diagnosis", {})
    dec_raw = data.get("decision", {})

    diagnosis = DiagnosisResult(
        request_kind=diag_raw.get("request_kind", "RESOUCE").upper(),
        student_stage=diag_raw.get("student_stage", "EARLY").upper(),
        student_state=diag_raw.get("student_state", "UNKNOWN").upper(),
        has_attempt=bool(diag_raw.get("has_attempt", False)),
        needs_diagnosis=bool(diag_raw.get("needs_diagnosis", True)),
        tool_context_known=bool(diag_raw.get("tool_context_known", False)),
        expertise_level=diag_raw.get("expertise_level", "UNKNOWN").upper(),
        frustration_level=diag_raw.get("frustration_level", "UNKNOWN").upper(),
        implementation_allowed=bool(diag_raw.get("implementation_allowed", False)),
        confidence=float(diag_raw.get("confidence", 0.0)),
        rationale=diag_raw.get("rationale", [])
    )

    support_level = dec_raw.get("support_level", "EXPLAIN").upper()
    decision = SupportDecision(
        support_level=support_level,
        response_prompt_file=RESPONSE_PROMPT_FILES.get(support_level, RESPONSE_PROMPT_FILES["DIAGNOSE"]),
        can_show_code=bool(dec_raw.get("can_show_code", False)),
        must_end_with_question=bool(dec_raw.get("must_end_with_question", True)),
        should_request_attempt=bool(dec_raw.get("should_request_attempt", True)),
        confidence=float(dec_raw.get("confidence", 0.0)),
        rationale=dec_raw.get("rationale", [])
    )
    return diagnosis, decision

async def diagnose_student(
    client,
    route: Dict[str, Any],
    llm_history: List[Dict[str, Any]],
    user_message: str,
) -> DiagnosisResult:
    system_prompt = "\n\n".join(
        [
            load_prompt(BASE_PROMPT_FILES["identity"]),
            load_prompt(_phase_prompt_file(route.get("phase"))),
            load_prompt(BASE_PROMPT_FILES["diagnose"]),
        ]
    )
    payload = _make_chain_context(route, llm_history, user_message)
    data = await _call_json(client, system_prompt, payload)
    return DiagnosisResult(
        request_kind=(data.get("request_kind") or "PRODUCT").upper(),
        student_stage=(data.get("student_stage") or "EARLY").upper(),
        student_state=(data.get("student_state") or "UNKNOWN").upper(),
        has_attempt=bool(data.get("has_attempt", False)),
        needs_diagnosis=bool(data.get("needs_diagnosis", True)),
        tool_context_known=bool(data.get("tool_context_known", False)),
        expertise_level=(data.get("expertise_level") or "UNKNOWN").upper(),
        frustration_level=(data.get("frustration_level") or "UNKNOWN").upper(),
        implementation_allowed=bool(data.get("implementation_allowed", False)),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        rationale=data.get("rationale", []) or [],
    )


async def choose_support_level(
    client,
    route: Dict[str, Any],
    diagnosis: DiagnosisResult,
    llm_history: List[Dict[str, Any]],
    user_message: str,
) -> SupportDecision:
    system_prompt = "\n\n".join(
        [
            load_prompt(BASE_PROMPT_FILES["identity"]),
            load_prompt(_phase_prompt_file(route.get("phase"))),
            load_prompt(BASE_PROMPT_FILES["decide_support"]),
        ]
    )
    payload = _make_chain_context(route, llm_history, user_message)
    payload += "\n\nDIAGNOSIS_JSON:\n" + json.dumps(
        diagnosis.__dict__, indent=2
    )

    data = await _call_json(client, system_prompt, payload)
    support_level = (data.get("support_level") or "DIAGNOSE").upper()

    logger.info(
        "Support decision: %s (confidence: %.2f)",
        support_level,
        float(data.get("confidence", 0.0) or 0.0),
    )

    response_prompt_file = RESPONSE_PROMPT_FILES.get(
        support_level, RESPONSE_PROMPT_FILES["DIAGNOSE"]
    )
    return SupportDecision(
        support_level=support_level,
        response_prompt_file=response_prompt_file,
        can_show_code=bool(data.get("can_show_code", False)),
        must_end_with_question=bool(data.get("must_end_with_question", True)),
        should_request_attempt=bool(data.get("should_request_attempt", True)),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        rationale=data.get("rationale", []) or [],
    )



async def generate_full_reply(    
    client,
    route: Dict[str, Any],
    diagnosis: DiagnosisResult,
    decision: SupportDecision,
    llm_history: List[Dict[str, Any]],
    user_message: str,) -> str:
    system_prompt = "\n\n".join([
        load_prompt(BASE_PROMPT_FILES["identity"]),
        load_prompt(_phase_prompt_file(route.get("phase"))),
        load_prompt(decision.response_prompt_file),
        load_prompt(BASE_PROMPT_FILES["file_handler"]),
    ])

    control_header = {
        "route": route,
        "diagnosis": diagnosis.__dict__,
        "decision": decision.__dict__,
    }

    user_payload = (
        "CONTROL_STATE:\n" + json.dumps(control_header, indent=2) + "\n\n"
        "RECENT_HISTORY:\n" + _compact_history(llm_history) + "\n\n"
        "CURRENT_USER_INPUT_WITH_FILES:\n" + user_message
    )

    resp = await client.chat.completions.create(
        model=CHAIN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload}
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content or ""


async def generate_reply_stream(
    client,
    route: Dict[str, Any],
    diagnosis: DiagnosisResult,
    decision: SupportDecision,
    llm_history: List[Dict[str, Any]],
    user_message: str,
):
    # Load prompts
    system_prompt = "\n\n".join([
        load_prompt(BASE_PROMPT_FILES["identity"]),
        load_prompt(_phase_prompt_file(route.get("phase"))),
        load_prompt(decision.response_prompt_file),
        load_prompt(BASE_PROMPT_FILES["file_handler"]),
    ])

    control_header = {
        "route": route,
        "diagnosis": diagnosis.__dict__,
        "decision": decision.__dict__,
    }

    user_payload = (
        "CONTROL_STATE:\n" + json.dumps(control_header, indent=2) + "\n\n"
        "RECENT_HISTORY:\n" + _compact_history(llm_history) + "\n\n"
        "CURRENT_USER_INPUT_WITH_FILES:\n" + user_message
    )

    logger.info("--- GENERATING STREAM ---")
    logger.info(f"Target Support Level: {decision.support_level}")
    logger.info(f"Phase Context: {route.get('phase')}")

    return await client.chat.completions.create(
        model=CHAIN_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload}
        ],
        temperature=0.3,
        stream=True
    )



async def check_reply(client, route, diagnosis, decision, draft_reply, llm_history, user_message) -> CheckResult:
    system_prompt = load_prompt(BASE_PROMPT_FILES["check_reply"])
    payload = {"diag": diagnosis.__dict__, "dec": decision.__dict__, "draft": draft_reply, "user": user_message}
    data = await _call_json(client, system_prompt, json.dumps(payload))
    return CheckResult(
        is_safe=bool(data.get("is_safe", False)),
        leaks_solution=bool(data.get("leaks_solution", True)),
        skipped_diagnosis=bool(data.get("skipped_diagnosis", False)),
        reason=data.get("reason", "unknown")
    )

async def rewrite_reply(client, route, diagnosis, decision, draft_reply, check, llm_history, user_message) -> str:
    system_prompt = "\n\n".join([
        load_prompt(BASE_PROMPT_FILES["identity"]),
        load_prompt(BASE_PROMPT_FILES["rewrite_reply"]),
    ])
    payload = {"draft": draft_reply, "reason": check.reason, "user": user_message}
    resp = await client.chat.completions.create(
        model=CHAIN_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": json.dumps(payload)}],
        temperature=0.1,
    )
    return resp.choices[0].message.content or ""


async def run_srl_chain(client, route, llm_history, user_message) -> tuple[str, DiagnosisResult, SupportDecision]:
    diagnosis, decision = await diagnose_and_decide(client, route, llm_history, user_message)
    draft_reply = await generate_full_reply(client, route, diagnosis, decision, llm_history, user_message)
    check = await check_reply(client, route, diagnosis, decision, draft_reply, llm_history, user_message)

    if not check.is_safe or check.leaks_solution:
        logger.info(f"Safety check failed: {check.reason}. Attempting rewrite...")
        final_reply = await rewrite_reply(client, route, diagnosis, decision, draft_reply, check, llm_history, user_message)
    else:
        final_reply = draft_reply

    return final_reply, diagnosis, decision