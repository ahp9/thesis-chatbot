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
    "identity": "base/srl_model_v1.txt",
    "phase_forethought": "phases/forethought_core.txt",
    "phase_performance": "phases/performance_core.txt",
    "phase_reflection": "phases/reflection_core.txt",
    "diagnose": "chains/diagnose_student_v1.txt",
    "decide_support": "chains/choose_support_level_v1.txt",
    "check_reply": "chains/check_solution_leak_v1.txt",
    "rewrite_reply": "chains/fallback_rewrite_v1.txt",
}

RESPONSE_PROMPT_FILES = {
    "DIAGNOSE": "responses/respond_diagnose.txt",
    "QUESTION": "responses/respond_question.txt",
    "HINT": "responses/respond_hint.txt",
    "STRUCTURE": "responses/respond_structure.txt",
    "EXPLAIN": "responses/respond_explain.txt",
    "PARTIAL": "responses/respond_partial.txt",
    "REFLECT": "responses/respond_reflect.txt",
    "EVALUATION": "responses/respond_evaluation.txt",
}


async def _call_json(client, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """Use direct instruction prompting and template-based prompting for machine-readable control.
    [Ch. 3.1.1, p. 38; Ch. 3.1.10, p. 47]"""
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
        logger.warning("JSON parse failed for control prompt: %s", content[:500])
        return {}


async def _call_text(client, system_prompt: str, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    resp = await client.chat.completions.create(
        model=CHAIN_MODEL,
        messages=[{"role": "system", "content": system_prompt}, *messages],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def _phase_prompt_file(phase: str) -> str:
    phase = (phase or "PERFORMANCE").upper()
    if phase == "FORETHOUGHT":
        return BASE_PROMPT_FILES["phase_forethought"]
    if phase == "REFLECTION":
        return BASE_PROMPT_FILES["phase_reflection"]
    return BASE_PROMPT_FILES["phase_performance"]


def _compact_history(llm_history: List[Dict[str, Any]], limit: int = 8) -> str:
    recent = llm_history[-limit:] if llm_history else []
    lines: List[str] = []
    for message in recent:
        role = (message.get("role") or "user").upper()
        content = (message.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(no prior context)"


def _make_chain_context(route: Dict[str, Any], llm_history: List[Dict[str, Any]], user_message: str) -> str:
    return (
        f"CURRENT_PHASE: {route.get('phase', 'PERFORMANCE')}\n"
        f"ROUTER_STRATEGY: {route.get('strategy', 'NONE')}\n"
        f"ROUTER_CONFIDENCE: {route.get('confidence', 0.0)}\n\n"
        f"RECENT_HISTORY:\n{_compact_history(llm_history)}\n\n"
        f"CURRENT_USER_MESSAGE:\n{user_message}"
    )


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
    payload += "\n\nDIAGNOSIS_JSON:\n" + json.dumps(diagnosis.__dict__, indent=2)
    
    data = await _call_json(client, system_prompt, payload)
    support_level = (data.get("support_level") or "DIAGNOSE").upper()
    
    logger.info("Support decision: %s (confidence: %.2f)", support_level, float(data.get("confidence", 0.0) or 0.0))
    
    response_prompt_file = RESPONSE_PROMPT_FILES.get(support_level, RESPONSE_PROMPT_FILES["DIAGNOSE"])
    return SupportDecision(
        support_level=support_level,
        response_prompt_file=response_prompt_file,
        can_show_code=bool(data.get("can_show_code", False)),
        must_end_with_question=bool(data.get("must_end_with_question", True)),
        should_request_attempt=bool(data.get("should_request_attempt", True)),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        rationale=data.get("rationale", []) or [],
    )


async def generate_reply(
    client,
    route: Dict[str, Any],
    diagnosis: DiagnosisResult,
    decision: SupportDecision,
    llm_history: List[Dict[str, Any]],
    user_message: str,
) -> str:
    # Template-based prompting uses small reusable prompts per response mode.
    # [Ch. 3.1.10, p. 47]
    # Conditional prompting adapts the answer to student state and progress.
    # [Ch. 3.2.9, p. 60]
    system_prompt = "\n\n".join(
        [
            load_prompt(BASE_PROMPT_FILES["identity"]),
            load_prompt(_phase_prompt_file(route.get("phase"))),
            load_prompt(decision.response_prompt_file),
        ]
    )

    control_header = {
        "route": route,
        "diagnosis": diagnosis.__dict__,
        "decision": decision.__dict__,
    }
    
    logger.info("Route: %s, Diagnosis: %s, Decision: %s", route, diagnosis, decision)
    
    messages = [
        {
            "role": "user",
            "content": (
                "CONTROL_STATE:\n"
                + json.dumps(control_header, indent=2)
                + "\n\n"
                + "RECENT_HISTORY:\n"
                + _compact_history(llm_history)
                + "\n\n"
                + "CURRENT_USER_MESSAGE:\n"
                + user_message
            ),
        }
    ]
    return await _call_text(client, system_prompt, messages, temperature=0.3)


async def check_reply(
    client,
    route: Dict[str, Any],
    diagnosis: DiagnosisResult,
    decision: SupportDecision,
    draft_reply: str,
    llm_history: List[Dict[str, Any]],
    user_message: str,
) -> CheckResult:
    system_prompt = load_prompt(BASE_PROMPT_FILES["check_reply"])
    payload = {
        "route": route,
        "diagnosis": diagnosis.__dict__,
        "decision": decision.__dict__,
        "recent_history": _compact_history(llm_history),
        "current_user_message": user_message,
        "draft_reply": draft_reply,
    }
    data = await _call_json(client, system_prompt, json.dumps(payload, indent=2))
    return CheckResult(
        is_safe=bool(data.get("is_safe", False)),
        leaks_solution=bool(data.get("leaks_solution", True)),
        skipped_diagnosis=bool(data.get("skipped_diagnosis", False)),
        reason=(data.get("reason") or "unknown").strip(),
    )


async def rewrite_reply(
    client,
    route: Dict[str, Any],
    diagnosis: DiagnosisResult,
    decision: SupportDecision,
    draft_reply: str,
    check: CheckResult,
    llm_history: List[Dict[str, Any]],
    user_message: str,
) -> str:
    system_prompt = "\n\n".join(
        [
            load_prompt(BASE_PROMPT_FILES["identity"]),
            load_prompt(_phase_prompt_file(route.get("phase"))),
            load_prompt(BASE_PROMPT_FILES["rewrite_reply"]),
        ]
    )
    payload = {
        "route": route,
        "diagnosis": diagnosis.__dict__,
        "decision": decision.__dict__,
        "check": check.__dict__,
        "recent_history": _compact_history(llm_history),
        "current_user_message": user_message,
        "draft_reply": draft_reply,
    }
    return await _call_text(client, system_prompt, [{"role": "user", "content": json.dumps(payload, indent=2)}], temperature=0.1)


async def run_srl_chain(
    client,
    route: Dict[str, Any],
    llm_history: List[Dict[str, Any]],
    user_message: str,
) -> Dict[str, Any]:
    diagnosis = await diagnose_student(client, route, llm_history, user_message)
    decision = await choose_support_level(client, route, diagnosis, llm_history, user_message)
    draft_reply = await generate_reply(client, route, diagnosis, decision, llm_history, user_message)
    check = await check_reply(client, route, diagnosis, decision, draft_reply, llm_history, user_message)

    final_reply = draft_reply
    if not check.is_safe or check.leaks_solution or check.skipped_diagnosis:
        final_reply = await rewrite_reply(
            client,
            route,
            diagnosis,
            decision,
            draft_reply,
            check,
            llm_history,
            user_message,
        )

    return {
        "reply": final_reply,
        "diagnosis": diagnosis.__dict__,
        "decision": decision.__dict__,
        "check": check.__dict__,
        "draft_reply": draft_reply,
    }