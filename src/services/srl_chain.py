import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from services.generation.generation import build_generation_context
from services.history_adapter import (
    build_learning_trajectory,
    last_assistant_reply,
    recent_control_state,
)
from services.policy.policy_config import response_prompt_file_for
from services.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

CONTROL_MODEL = "gpt-4.1-mini"     # diagnosis + support decision
GENERATION_MODEL = "gpt-4.1-mini"   # reply writing
CHECK_MODEL = "gpt-4o-mini"        # safety check
REWRITE_MODEL = "gpt-4o-mini"      # rewrite if needed

# Support levels where a safety/leak check is actually meaningful.
SAFETY_CHECK_LEVELS = {"PARTIAL", "EXPLAIN", "STRUCTURE", "EVALUATION"}

# Dataclasses
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
class CheckResult:
    is_safe: bool
    leaks_solution: bool
    skipped_diagnosis: bool
    reason: str
    was_skipped: bool = False


# Prompt file maps
BASE_PROMPT_FILES = {
    "identity": "base/srl_model_v4.txt",
    "generation_style": "base/srl_respond_tone.txt",
    "phase_forethought": "phases/forethought_core_v3.txt",
    "phase_performance": "phases/performance_core_v2.txt",
    "phase_reflection": "phases/reflection_core_v2.txt",
    "check_reply": "chains/check_solution_leak_v3.txt",
    "rewrite_reply": "chains/fallback_rewrite_v3.txt",
    "checkpoint_and_decide": "chains/student_state_v6.txt",
    "file_handler": "base/file.txt",
}


# JSON parsing
def _extract_json(raw: str) -> Dict[str, Any]:
    """
    Try to extract a JSON object from the model's raw response.

    Handles three common failure modes:
      1. Model wraps JSON in ```json ... ``` fences
      2. Model prepends a line of prose before the JSON
      3. Model returns valid JSON directly

    Returns an empty dict only if all attempts fail.
    """
    if not raw or not raw.strip():
        logger.warning("_extract_json: model returned empty response")
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
        "_extract_json: could not parse model response. Raw content (first 500 chars):\n%s",
        raw[:500],
    )
    return {}


async def _call_json(
    client, system_prompt: str, user_prompt: str, model: str = CONTROL_MODEL
) -> Tuple[str, Dict[str, Any], bool]:
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
        role = (m.get('role') or 'user').upper()
        content = (m.get('content') or '').strip()
        if content:
            lines.append(f"{role}: {content[:500]}")
    return "\n".join(lines) if lines else "(no prior context)"


def _build_native_history(
    llm_history: List[Dict[str, Any]],
    limit: int = 8,
) -> List[Dict[str, str]]:
    """
    Build a proper OpenAI-style alternating user/assistant message list.
    Used for generate_full_reply so the model sees real conversation turns.
    File block content from old turns is stripped to avoid token bloat.
    """
    clean: List[Dict[str, str]] = []
    for m in llm_history[-(limit * 2):]:
        role = m.get("role", "")
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

    while clean and clean[0]["role"] == "assistant":
        clean.pop(0)

    return clean


def _build_checkpoint_payload(
    route: Dict[str, Any],
    llm_history: List[Dict[str, Any]],
    user_message: str,
) -> str:
    """
    Build the input payload for the checkpoint+decide stage.
    """
    recent_control = recent_control_state(llm_history)
    trajectory = build_learning_trajectory(llm_history, limit=3)

    parts = [
        f"CURRENT_PHASE:\n{route.get('phase', 'UNKNOWN')}",
        f"LEARNING_TRAJECTORY (last 3 turns):\n{trajectory}",
        f"PREVIOUS_CONTROL:\n{json.dumps(recent_control, indent=2)}",
        f"RECENT_HISTORY:\n{_compact_history(llm_history)}",
        f"CURRENT_USER_MESSAGE:\n{user_message}",
    ]
    return "\n\n".join(parts)


def _should_run_safety_check(decision: SupportDecision) -> bool:
    """
    Run the check when the support level can plausibly produce a leaked
    solution or runnable code. Skip for structurally safe levels.
    """
    if decision.can_show_code:
        return True
    if decision.support_level in SAFETY_CHECK_LEVELS:
        return True
    return False

def _strip_plan_block(text: str) -> str:
    """Remove <plan>...</plan> block from generation output before returning."""
    return re.sub(r"<plan>.*?</plan>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Fallback values — used ONLY when parsing genuinely fails.
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
    Run the combined checkpoint + decision stage.
    """
    prompt_parts = [
        load_prompt(BASE_PROMPT_FILES["identity"]),
        load_prompt(_phase_prompt_file(route.get("phase"))),
        load_prompt(BASE_PROMPT_FILES["checkpoint_and_decide"]),
    ]
    if _has_file_content(user_message):
        prompt_parts.append(load_prompt(BASE_PROMPT_FILES["file_handler"]))

    system_prompt = "\n\n".join(prompt_parts)
    payload = _build_checkpoint_payload(route, llm_history, user_message)
    raw_text, data, parse_ok = await _call_json(
        client, system_prompt, payload, model=CONTROL_MODEL
    )

    debug = {
        "raw_text": raw_text,
        "parsed_json": data,
        "parse_ok": parse_ok,
        "fallback_used": False,
    }

    if not parse_ok:
        logger.warning(
            "checkpoint_and_decide: JSON parse failed — using fallback values."
        )
        debug["fallback_used"] = True
        return _fallback_checkpoint(), _fallback_decision(), debug

    checkp_raw = data.get("checkpoint", {})
    dec_raw = data.get("decision", {})

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


def _build_writer_brief(route, checkpoint, decision) -> dict:
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
    Generate the tutor reply using native multi-turn history.

    The system prompt is built in layers:
      1. SRL identity (srl_model_v4)
      2. Phase prompt (forethought / performance / reflection)
      3. Support-level response prompt (respond_X)
      4. Generation context: depth + affective + coherence + expansion instructions
         These were previously commented out. They are now active and also carry
         expertise_level so the writer has an explicit anti-parroting contract
         for INTERMEDIATE/ADVANCED students.
      5. File handler (conditional on file content in message)

    History is passed as real alternating user/assistant turns rather than
    a serialised string, which reduces token use and improves coherence.
    """
    recent_control = recent_control_state(llm_history)
    previous_support_level = recent_control.get("previous_support_level")

    # Build the dynamic instruction block from generation.py.
    # expertise_level is now passed so the writer receives an explicit tone
    # contract: INTERMEDIATE students must not have their own words parroted
    # back at them; ADVANCED students skip orientation entirely.
    generation_context = build_generation_context(
        support_depth=decision.support_depth,
        frustration_level=checkpoint.frustration_level,
        support_level=decision.support_level,
        phase=route.get("phase", "PERFORMANCE"),
        previous_support_level=previous_support_level,
        expertise_level=checkpoint.expertise_level,
    )
    
    prompt_parts = []

    prompt_parts.extend([
        load_prompt(BASE_PROMPT_FILES["identity"]),
        load_prompt(_phase_prompt_file(route.get("phase"))),
        generation_context,
        load_prompt(decision.response_prompt_file),
    ])
    
    if gate_hint:
        logger.info("Gate hint provided: %s")
        prompt_parts.append(gate_hint)
    
    if _has_file_content(user_message):
        prompt_parts.append(load_prompt(BASE_PROMPT_FILES["file_handler"]))

    system_prompt = "\n\n".join(prompt_parts)

    # Slim brief — only what a writer needs
    writer_brief = _build_writer_brief(route, checkpoint, decision)
    previous_reply = last_assistant_reply(llm_history)

    current_turn_content = (
        f"TUTOR_BRIEF:\n{json.dumps(writer_brief, indent=2)}\n\n"
        f"PREVIOUS_REPLY:\n{previous_reply}\n\n"
        f"CURRENT_USER_INPUT_WITH_FILES:\n{user_message}"
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
        "chk": checkpoint.__dict__,
        "dec": decision.__dict__,
        "draft": draft_reply,
        "user": user_message,
        "recent_history": _compact_history(llm_history, limit=4),
        "route": route,
    }
    raw_text, data, parse_ok = await _call_json(
        client,
        system_prompt,
        json.dumps(payload),
        model=CHECK_MODEL,
    )

    if not parse_ok:
        logger.warning("check_reply: JSON parse failed — defaulting to safe=False")
        logger.warning("check_reply: JSON parse failed. raw_text=%s", raw_text)
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
    system_prompt = "\n\n".join([
        load_prompt(BASE_PROMPT_FILES["identity"]),
        load_prompt(BASE_PROMPT_FILES["rewrite_reply"]),
    ])

    recent_control = recent_control_state(llm_history)
    previous_reply = last_assistant_reply(llm_history)

    payload = {
        "draft": draft_reply,
        "reason": check.reason,
        "user": user_message,
        "recent_history": _compact_history(llm_history, limit=4),
        "route": route,
        "checkpoint": checkpoint.__dict__,
        "decision": decision.__dict__,
        "previous_control": recent_control,
        "previous_reply": previous_reply,
    }
    resp = await client.chat.completions.create(
        model=REWRITE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload)},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Public entry point
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
        logger.info("Safety check failed: %s. Rewriting...", check.reason)
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
    else:
        final_reply = draft_reply

    return {
        "reply": final_reply,
        "draft_reply": draft_reply,
        "diagnosis": diagnosis.__dict__,
        "decision": decision.__dict__,
        "check": check.__dict__,
        "was_rewritten": final_reply != draft_reply,
    }