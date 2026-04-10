import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from services.generation.generation import (
    build_filled_structure,
    get_coherence_instruction,
)
from services.history_adapter import (
    build_learning_trajectory,
    last_assistant_reply,
    recent_control_state,
)
from services.policy.policy_config import response_prompt_file_for
from services.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

CONTROL_MODEL    = "gpt-4.1-mini"   # checkpoint + support decision
PLAN_MODEL       = "gpt-4o-mini"    # reply planner  (cheap, ~200 token output)
GENERATION_MODEL = "gpt-4.1-mini"   # reply writer
CHECK_MODEL      = "gpt-4o-mini"    # safety check
REWRITE_MODEL    = "gpt-4o-mini"    # rewrite if safety check fails

# Support levels where a safety/leak check is meaningful.
SAFETY_CHECK_LEVELS = {"PARTIAL", "EXPLAIN", "STRUCTURE", "EVALUATION"}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CheckpointResult:
    request_kind: str
    task_stage: str
    progress_state: str
    has_attempt: bool
    context_gap: str
    expertise_level: str
    frustration_level: str
    srl_focus: str
    implementation_allowed: bool
    confidence: float
    rationale: List[str]
    parse_ok: bool = True


@dataclass
class SupportDecision:
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
class ReplyPlan:
    """
    Output of the planner call.

    student_has      — what the student demonstrably understands from their message
    gap              — the one tension or decision they have not yet addressed
    move1_aim        — cognitive aim of Move 1 (not a format description)
    move1_depth      — brief | substantive | rich
    handback_type    — question | action | partial_frame
    handback_content — the actual question, action, or partial frame (specific)
    off_limits       — the specific content that would constitute a leaked answer
    tone_note        — opening register note if frustration/phase requires one
    parse_ok         — False if planner JSON failed to parse
    """
    student_has:      str
    gap:              str
    move1_aim:        str
    move1_depth:      str
    handback_type:    str
    handback_content: str
    off_limits:       str
    tone_note:        str
    parse_ok:         bool = True


@dataclass
class CheckResult:
    is_safe: bool
    leaks_solution: bool
    skipped_diagnosis: bool
    reason: str
    was_skipped: bool = False


# ---------------------------------------------------------------------------
# Prompt file registry
# ---------------------------------------------------------------------------

BASE_PROMPT_FILES = {
    # Structure / identity
    "tutor_structure":       "base/tutor_structure.txt",

    # Phase cognitive guidance (no structural instructions)
    "phase_forethought":     "phases/forethought_core_v3.txt",
    "phase_performance":     "phases/performance_core_v2.txt",
    "phase_reflection":      "phases/reflection_core_v1.txt",

    # Chain prompts
    "reply_planner":         "chains/reply_planner.txt",
    "check_reply":           "chains/check_solution_leak_v3.txt",
    "rewrite_reply":         "chains/fallback_rewrite_v3.txt",
    "checkpoint_and_decide": "chains/student_state_v6.txt",

    # Conditional
    "file_handler":          "base/file.txt",
}


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> Dict[str, Any]:
    """
    Extract a JSON object from the model's raw response.

    Handles:
      1. Valid JSON returned directly
      2. JSON wrapped in ```json ... ``` fences
      3. Prose before the JSON object

    Returns an empty dict only if all attempts fail.
    """
    if not raw or not raw.strip():
        logger.warning("_extract_json: empty response from model")
        return {}

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    logger.error(
        "_extract_json: could not parse model response. First 500 chars:\n%s",
        raw[:500],
    )
    return {}


async def _call_json(
    client,
    system_prompt: str,
    user_prompt: str,
    model: str = CONTROL_MODEL,
) -> Tuple[str, Dict[str, Any], bool]:
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    data = _extract_json(raw)
    return raw, data, bool(data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _phase_prompt_file(phase: Optional[str]) -> str:
    phase = (phase or "PERFORMANCE").upper()
    if phase == "FORETHOUGHT":
        return BASE_PROMPT_FILES["phase_forethought"]
    if phase == "REFLECTION":
        return BASE_PROMPT_FILES["phase_reflection"]
    return BASE_PROMPT_FILES["phase_performance"]


def _has_file_content(user_message: str) -> bool:
    markers = ["FILE:", "FILE_BLOCK:", "FILES:", "CURRENT_USER_INPUT_WITH_FILES:"]
    return any(marker in user_message for marker in markers)


def _compact_history(llm_history: List[Dict[str, Any]], limit: int = 8) -> str:
    recent = llm_history[-limit:] if llm_history else []
    lines = []
    for m in recent:
        role    = (m.get("role") or "user").upper()
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content[:500]}")
    return "\n".join(lines) if lines else "(no prior context)"


def _build_native_history(
    llm_history: List[Dict[str, Any]],
    limit: int = 8,
) -> List[Dict[str, str]]:
    """
    Build an OpenAI-style alternating user/assistant message list.
    File block content from older turns is stripped to avoid token bloat.
    """
    clean: List[Dict[str, str]] = []
    for m in llm_history[-(limit * 2):]:
        role    = m.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if "--- FILE:" in content:
            content = content.split("--- FILE:")[0].strip()
            if not content:
                content = "[previous file upload]"
        clean.append({"role": role, "content": content})

    clean = clean[-limit:]

    # Ensure the list starts with a user turn
    while clean and clean[0]["role"] == "assistant":
        clean.pop(0)

    return clean


def _build_checkpoint_payload(
    route: Dict[str, Any],
    llm_history: List[Dict[str, Any]],
    user_message: str,
) -> str:
    recent_control = recent_control_state(llm_history)
    trajectory     = build_learning_trajectory(llm_history, limit=3)

    parts = [
        f"CURRENT_PHASE:\n{route.get('phase', 'UNKNOWN')}",
        f"LEARNING_TRAJECTORY (last 3 turns):\n{trajectory}",
        f"PREVIOUS_CONTROL:\n{json.dumps(recent_control, indent=2)}",
        f"RECENT_HISTORY:\n{_compact_history(llm_history)}",
        f"CURRENT_USER_MESSAGE:\n{user_message}",
    ]
    return "\n\n".join(parts)


def _should_run_safety_check(decision: SupportDecision) -> bool:
    if decision.can_show_code:
        return True
    if decision.support_level in SAFETY_CHECK_LEVELS:
        return True
    return False


def _strip_plan_block(text: str) -> str:
    """Remove any <plan>...</plan> block the writer may have left in its output."""
    return re.sub(r"<plan>.*?</plan>", "", text, flags=re.DOTALL).strip()


def _build_writer_brief(
    route: Dict[str, Any],
    checkpoint: CheckpointResult,
    decision: SupportDecision,
) -> dict:
    return {
        "phase":                  route.get("phase", "PERFORMANCE"),
        "support_level":          decision.support_level,
        "support_depth":          decision.support_depth,
        "can_show_code":          decision.can_show_code,
        "must_end_with_question": decision.must_end_with_question,
        "expertise_level":        checkpoint.expertise_level,
        "frustration_level":      checkpoint.frustration_level,
        "srl_focus":              checkpoint.srl_focus,
        "has_attempt":            checkpoint.has_attempt,
    }


# ---------------------------------------------------------------------------
# Fallback values — used only when LLM parsing genuinely fails
# ---------------------------------------------------------------------------

def _fallback_checkpoint() -> CheckpointResult:
    return CheckpointResult(
        request_kind="PRODUCT",
        task_stage="WORKING",
        progress_state="MOVING",
        has_attempt=False,
        context_gap="SMALL",
        expertise_level="NOVICE",
        frustration_level="LOW",
        srl_focus="STRATEGY",
        implementation_allowed=False,
        confidence=0.0,
        rationale=["PARSE_FAILED — fallback values in use"],
        parse_ok=False,
    )


def _fallback_decision() -> SupportDecision:
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


def _fallback_plan() -> ReplyPlan:
    return ReplyPlan(
        student_has="(planner parse failed — engaging with student's situation)",
        gap="(planner parse failed — targeting the next unaddressed step)",
        move1_aim="engage specifically with the student's message and name the relevant tension",
        move1_depth="substantive",
        handback_type="question",
        handback_content="What have you tried so far, and where did it stop making sense?",
        off_limits="do not state the final answer or complete the task",
        tone_note="",
        parse_ok=False,
    )


# ---------------------------------------------------------------------------
# Chain steps
# ---------------------------------------------------------------------------

async def checkpoint_and_decide(
    client,
    route: Dict[str, Any],
    llm_history: List[Dict[str, Any]],
    user_message: str,
) -> Tuple[CheckpointResult, SupportDecision, dict]:
    """
    Classify the student's state and decide the support level.

    This is a classification task — it operates on the student's state
    (phase, trajectory, progress, frustration) and produces labels.
    It does NOT reason about the specific content of the reply.
    """
    prompt_parts = [
        load_prompt(BASE_PROMPT_FILES["checkpoint_and_decide"]),
    ]
    if _has_file_content(user_message):
        prompt_parts.append(load_prompt(BASE_PROMPT_FILES["file_handler"]))

    system_prompt = "\n\n".join(prompt_parts)
    payload       = _build_checkpoint_payload(route, llm_history, user_message)

    raw_text, data, parse_ok = await _call_json(
        client, system_prompt, payload, model=CONTROL_MODEL
    )

    debug = {
        "raw_text":      raw_text,
        "parsed_json":   data,
        "parse_ok":      parse_ok,
        "fallback_used": False,
    }

    if not parse_ok:
        logger.warning("checkpoint_and_decide: JSON parse failed — using fallback values.")
        debug["fallback_used"] = True
        return _fallback_checkpoint(), _fallback_decision(), debug

    checkp_raw = data.get("checkpoint", {})
    dec_raw    = data.get("decision", {})

    diagnosis = CheckpointResult(
        request_kind=(checkp_raw.get("request_kind") or "PRODUCT").upper(),
        task_stage=(checkp_raw.get("task_stage") or "WORKING").upper(),
        progress_state=(checkp_raw.get("progress_state") or "MOVING").upper(),
        has_attempt=bool(checkp_raw.get("has_attempt", False)),
        context_gap=(checkp_raw.get("context_gap") or "NONE").upper(),
        expertise_level=(
            checkp_raw.get("expertise_level")
            or ("UNKNOWN" if not llm_history else "INTERMEDIATE")
        ).upper(),
        frustration_level=(checkp_raw.get("frustration_level") or "LOW").upper(),
        srl_focus=(
            checkp_raw.get("srl_focus")
            or ("STRATEGY" if checkp_raw.get("request_kind") == "PRODUCT" else "NONE")
        ).upper(),
        implementation_allowed=bool(checkp_raw.get("implementation_allowed", False)),
        confidence=float(checkp_raw.get("confidence") or 0.0),
        rationale=checkp_raw.get("rationale") or [],
        parse_ok=True,
    )

    support_level = (dec_raw.get("support_level") or "QUESTION").upper()
    decision = SupportDecision(
        support_level=support_level,
        response_prompt_file=response_prompt_file_for(support_level),
        can_show_code=bool(dec_raw.get("can_show_code", False)),
        must_end_with_question=bool(dec_raw.get("must_end_with_question", True)),
        should_request_attempt=bool(dec_raw.get("should_request_attempt", False)),
        confidence=float(dec_raw.get("confidence") or 0.0),
        rationale=dec_raw.get("rationale") or [],
        support_depth=(dec_raw.get("support_depth") or "SUBSTANTIVE").upper(),
        parse_ok=bool(dec_raw),
    )

    return diagnosis, decision, debug


async def _plan_reply(
    client,
    route: Dict[str, Any],
    checkpoint: CheckpointResult,
    decision: SupportDecision,
    llm_history: List[Dict[str, Any]],
    user_message: str,
) -> ReplyPlan:
    # Build the filled structure document — this is the planner's
    # primary authority on how the reply should be shaped.
    filled_structure = build_filled_structure(
        expertise_level=checkpoint.expertise_level,
        phase=route.get("phase", "PERFORMANCE"),
        srl_focus=checkpoint.srl_focus,
        frustration_level=checkpoint.frustration_level,
        support_depth=decision.support_depth,
    )

    system_prompt = "\n\n".join([
        filled_structure,
        load_prompt(BASE_PROMPT_FILES["reply_planner"]),
    ])

    # Coherence instruction: computed from the delta between current and
    # previous support levels. Informs the planner's move1_aim so the
    # plan does not repeat the previous approach.
    coherence = get_coherence_instruction(
        current_support_level=decision.support_level,
        previous_support_level=recent_control_state(llm_history).get("previous_support_level"),
    )

    writer_brief   = _build_writer_brief(route, checkpoint, decision)
    previous_reply = last_assistant_reply(llm_history)

    payload = (
        f"WRITER_BRIEF:\n{json.dumps(writer_brief, indent=2)}\n\n"
        f"COHERENCE_NOTE:\n{coherence}\n\n"
        f"PREVIOUS_REPLY:\n{previous_reply}\n\n"
        f"STUDENT_MESSAGE:\n{user_message}"
    )

    raw, data, parse_ok = await _call_json(
        client, system_prompt, payload, model=PLAN_MODEL
    )

    if not parse_ok:
        logger.warning("_plan_reply: planner JSON parse failed — using fallback plan.")
        return _fallback_plan()

    return ReplyPlan(
        student_has=      (data.get("student_has")      or "").strip(),
        gap=              (data.get("gap")               or "").strip(),
        move1_aim=        (data.get("move1_aim")         or "").strip(),
        move1_depth=      (data.get("move1_depth")       or "substantive").strip().lower(),
        handback_type=    (data.get("handback_type")     or "question").strip().lower(),
        handback_content= (data.get("handback_content")  or "").strip(),
        off_limits=       (data.get("off_limits")        or "").strip(),
        tone_note=        (data.get("tone_note")         or "").strip(),
        parse_ok=True,
    )


async def _write_reply(
    client,
    route: Dict[str, Any],
    checkpoint: CheckpointResult,
    decision: SupportDecision,
    plan: ReplyPlan,
    llm_history: List[Dict[str, Any]],
    user_message: str,
    gate_hint: Optional[str] = None,
) -> str:
    """
    Call 2 of the two-call generation pipeline.

    The writer receives the concrete plan from the planner and executes it.
    It has clear, non-competing authority:
      - Phase prompt:    cognitive priorities and forbidden moves for this phase
      - respond_X file:  content character of Move 1 for this support mode

    The writer does NOT receive tutor_structure.txt directly — the structural
    contract was consumed by the planner and is now expressed in the plan.
    The plan tells the writer what to do; the phase and respond files tell it
    how to do it for this phase and mode.
    """
    prompt_parts = [
        load_prompt(_phase_prompt_file(route.get("phase"))),
        load_prompt(decision.response_prompt_file),
    ]

    if gate_hint:
        logger.info("Gate hint active — prepending first-turn gate prompt.")
        prompt_parts.append(gate_hint)

    if _has_file_content(user_message):
        prompt_parts.append(load_prompt(BASE_PROMPT_FILES["file_handler"]))

    system_prompt = "\n\n".join(prompt_parts)

    # Format the plan as a clear instruction block for the writer.
    # Each field is labelled so the writer knows exactly what to do
    # without re-reading structural documents.
    plan_block = "\n".join([
        "REPLY_PLAN",
        "----------",
        f"What the student has:  {plan.student_has}",
        f"Gap to address:        {plan.gap}",
        f"Move 1 aim:            {plan.move1_aim}",
        f"Move 1 depth:          {plan.move1_depth}",
        f"Handback type:         {plan.handback_type}",
        f"Handback content:      {plan.handback_content}",
        f"Off limits:            {plan.off_limits}",
        f"Tone note:             {plan.tone_note or 'none'}",
    ])

    current_turn_content = (
        f"{plan_block}\n\n"
        f"STUDENT_MESSAGE:\n{user_message}"
    )

    history_turns = _build_native_history(llm_history)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_turns)
    messages.append({"role": "user", "content": current_turn_content})

    resp = await client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=0.3,
    )
    raw = resp.choices[0].message.content or ""
    return _strip_plan_block(raw)


async def generate_full_reply(
    client,
    route: Dict[str, Any],
    checkpoint: CheckpointResult,
    decision: SupportDecision,
    llm_history: List[Dict[str, Any]],
    user_message: str,
    gate_hint: Optional[str] = None,
) -> str:
    """
    Single-call generation pipeline.

    The writer receives tutor_structure.txt (filled) directly, along with
    the phase prompt and respond_X file. It self-plans before writing.
    """
    filled_structure = build_filled_structure(
        expertise_level=checkpoint.expertise_level,
        phase=route.get("phase", "PERFORMANCE"),
        srl_focus=checkpoint.srl_focus,
        frustration_level=checkpoint.frustration_level,
        support_depth=decision.support_depth,
    )

    prompt_parts = [
        filled_structure,
        load_prompt(_phase_prompt_file(route.get("phase"))),
        load_prompt(decision.response_prompt_file),
    ]

    if gate_hint:
        logger.info("Gate hint active — prepending first-turn gate prompt.")
        prompt_parts.append(gate_hint)

    if _has_file_content(user_message):
        prompt_parts.append(load_prompt(BASE_PROMPT_FILES["file_handler"]))

    system_prompt = "\n\n".join(prompt_parts)

    coherence = get_coherence_instruction(
        current_support_level=decision.support_level,
        previous_support_level=recent_control_state(llm_history).get("previous_support_level"),
    )

    writer_brief   = _build_writer_brief(route, checkpoint, decision)
    previous_reply = last_assistant_reply(llm_history)

    current_turn_content = (
        f"WRITER_BRIEF:\n{json.dumps(writer_brief, indent=2)}\n\n"
        f"COHERENCE_NOTE:\n{coherence}\n\n"
        f"PREVIOUS_REPLY:\n{previous_reply}\n\n"
        f"STUDENT_MESSAGE:\n{user_message}"
    )

    history_turns = _build_native_history(llm_history)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_turns)
    messages.append({"role": "user", "content": current_turn_content})

    resp = await client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=0.3,
    )
    raw = resp.choices[0].message.content or ""
    return _strip_plan_block(raw)
# ---------------------------------------------------------------------------
# Safety check and rewrite
# ---------------------------------------------------------------------------

async def check_reply(
    client,
    route: Dict[str, Any],
    checkpoint: CheckpointResult,
    decision: SupportDecision,
    draft_reply: str,
    llm_history: List[Dict[str, Any]],
    user_message: str,
) -> CheckResult:
    """
    Run the safety/leak check only when the support level warrants it.
    Structurally safe levels (QUESTION, HINT, CLARIFY, REFLECT) are skipped.
    """
    if not _should_run_safety_check(decision):
        logger.info(
            "Safety check skipped: support_level=%s can_show_code=%s",
            decision.support_level,
            decision.can_show_code,
        )
        return CheckResult(
            is_safe=True,
            leaks_solution=False,
            skipped_diagnosis=False,
            reason="check bypassed — low-risk support level",
            was_skipped=True,
        )

    system_prompt = load_prompt(BASE_PROMPT_FILES["check_reply"])
    payload = {
        "chk":            checkpoint.__dict__,
        "dec":            decision.__dict__,
        "draft":          draft_reply,
        "user":           user_message,
        "recent_history": _compact_history(llm_history, limit=4),
        "route":          route,
    }

    raw_text, data, parse_ok = await _call_json(
        client,
        system_prompt,
        json.dumps(payload),
        model=CHECK_MODEL,
    )

    if not parse_ok:
        logger.warning("check_reply: JSON parse failed — defaulting to safe=False")
        logger.warning("check_reply raw_text: %s", raw_text)
        return CheckResult(
            is_safe=False,
            leaks_solution=True,
            skipped_diagnosis=False,
            reason="check parse failed — conservative rewrite triggered",
            was_skipped=False,
        )

    return CheckResult(
        is_safe=bool(data.get("is_safe", False)),
        leaks_solution=bool(data.get("leaks_solution", True)),
        skipped_diagnosis=bool(data.get("skipped_diagnosis", False)),
        reason=data.get("reason", "unknown"),
        was_skipped=False,
    )


async def rewrite_reply(
    client,
    route: Dict[str, Any],
    checkpoint: CheckpointResult,
    decision: SupportDecision,
    draft_reply: str,
    check: CheckResult,
    llm_history: List[Dict[str, Any]],
    user_message: str,
) -> str:
    """
    Rewrite a reply that failed the safety check.
    Uses the identity (tutor_structure filled) + fallback_rewrite prompt.
    """
    filled_structure = build_filled_structure(
        expertise_level=checkpoint.expertise_level,
        phase=route.get("phase", "PERFORMANCE"),
        srl_focus=checkpoint.srl_focus,
        frustration_level=checkpoint.frustration_level,
        support_depth=decision.support_depth,
    )

    system_prompt = "\n\n".join([
        filled_structure,
        load_prompt(BASE_PROMPT_FILES["rewrite_reply"]),
    ])

    recent_control = recent_control_state(llm_history)
    previous_reply = last_assistant_reply(llm_history)

    payload = {
        "draft":            draft_reply,
        "reason":           check.reason,
        "user":             user_message,
        "recent_history":   _compact_history(llm_history, limit=4),
        "route":            route,
        "checkpoint":       checkpoint.__dict__,
        "decision":         decision.__dict__,
        "previous_control": recent_control,
        "previous_reply":   previous_reply,
    }

    resp = await client.chat.completions.create(
        model=REWRITE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": json.dumps(payload)},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Public entry point (used by run_srl_chain legacy path)
# ---------------------------------------------------------------------------

async def run_srl_chain(
    client,
    route: Dict[str, Any],
    llm_history: List[Dict[str, Any]],
    user_message: str,
) -> Dict[str, Any]:

    diagnosis, decision, _ = await checkpoint_and_decide(
        client, route, llm_history, user_message
    )

    draft_reply = await generate_full_reply(
        client, route, diagnosis, decision, llm_history, user_message
    )

    check = await check_reply(
        client, route, diagnosis, decision, draft_reply, llm_history, user_message
    )

    if not check.is_safe or check.leaks_solution:
        logger.info("Safety check failed: %s — rewriting.", check.reason)
        final_reply = await rewrite_reply(
            client, route, diagnosis, decision,
            draft_reply, check, llm_history, user_message,
        )
    else:
        final_reply = draft_reply

    return {
        "reply":        final_reply,
        "draft_reply":  draft_reply,
        "diagnosis":    diagnosis.__dict__,
        "decision":     decision.__dict__,
        "check":        check.__dict__,
        "was_rewritten": final_reply != draft_reply,
    }