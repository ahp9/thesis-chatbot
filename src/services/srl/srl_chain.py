import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from services.generation.generation import build_filled_structure, get_coherence_instruction
from utils.history_adapter import last_assistant_reply, recent_control_state
from utils.prompt_loader import load_prompt
from services.policy.policy_config import response_prompt_file_for
from services.srl.srl_helpers import (
    build_checkpoint_payload,
    build_native_history,
    build_writer_brief,
    call_json,
    compact_history,
    file_referenced_but_missing,
    has_file_content,
    phase_prompt_file,
    should_run_safety_check,
    strip_plan_block,
)
from services.srl.srl_models import (
    BASE_PROMPT_FILES,
    CHECK_MODEL,
    CONTROL_MODEL,
    GENERATION_MODEL,
    REWRITE_MODEL,
    CheckpointResult,
    CheckResult,
    SupportDecision,
    _fallback_checkpoint,
    _fallback_decision,
)

logger = logging.getLogger(__name__)


async def checkpoint_and_decide(
    client,
    route: Dict[str, Any],
    llm_history: List[Dict[str, Any]],
    user_message: str,
) -> Tuple[CheckpointResult, SupportDecision, dict]:
    """Run the checkpoint LLM call and return a typed diagnosis, support decision, and debug dict."""
    prompt_parts = [load_prompt(BASE_PROMPT_FILES["checkpoint_and_decide"])]
    if has_file_content(user_message):
        prompt_parts.append(load_prompt(BASE_PROMPT_FILES["file_handler"]))

    system_prompt = "\n\n".join(prompt_parts)
    payload       = build_checkpoint_payload(route, llm_history, user_message)

    raw_text, data, parse_ok = await call_json(
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
        subtask_scope=(checkp_raw.get("subtask_scope") or "NEW_TASK").upper(),
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


async def generate_full_reply(
    client,
    route: Dict[str, Any],
    checkpoint: CheckpointResult,
    decision: SupportDecision,
    llm_history: List[Dict[str, Any]],
    user_message: str,
    guard: Optional[str] = None,
) -> str:
    """Generate the tutor reply for the current turn, handling missing files and guard hints."""
    missing_filename = file_referenced_but_missing(user_message, llm_history)
    if missing_filename:
        logger.info("File missing from context: %s — triggering missing file response.", missing_filename)
        missing_prompt = load_prompt(BASE_PROMPT_FILES["missing_file"])
        resp = await client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": missing_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content or ""

    filled_structure = build_filled_structure(
        expertise_level=checkpoint.expertise_level,
        phase=route.get("phase", "PERFORMANCE"),
        srl_focus=checkpoint.srl_focus,
        frustration_level=checkpoint.frustration_level,
        support_depth=decision.support_depth,
    )

    prompt_parts = [
        filled_structure,
        load_prompt(phase_prompt_file(route.get("phase"))),
        load_prompt(decision.response_prompt_file),
    ]

    if guard:
        logger.info("Guard hint active — prepending first-turn guard prompt.")
        prompt_parts.append(guard)

    if has_file_content(user_message):
        prompt_parts.append(load_prompt(BASE_PROMPT_FILES["file_handler"]))

    system_prompt = "\n\n".join(prompt_parts)

    coherence = get_coherence_instruction(
        current_support_level=decision.support_level,
        previous_support_level=recent_control_state(llm_history).get("previous_support_level"),
    )

    writer_brief   = build_writer_brief(route, checkpoint, decision)
    previous_reply = last_assistant_reply(llm_history)

    subtask_note = (
        "SUBTASK_SCOPE: NEW_TASK - treat this as a fresh question. "
        "The session context (dataset, assignment) is shared background, "
        "but the previous question's gap, variable choices, and reasoning "
        "do not apply here. Do not reference or continue prior local reasoning "
        "unless the student explicitly invokes it."
        if checkpoint.subtask_scope == "NEW_TASK"
        else "SUBTASK_SCOPE: CONTINUE - the student is explicitly continuing "
        "from the previous turn. Prior reasoning and context are relevant."
    )

    current_turn_content = (
        f"WRITER_BRIEF:\n{json.dumps(writer_brief, indent=2)}\n\n"
        f"SUBTASK_NOTE:\n{subtask_note}\n\n"
        f"COHERENCE_NOTE:\n{coherence}\n\n"
        f"PREVIOUS_REPLY:\n{previous_reply}\n\n"
        f"STUDENT_MESSAGE:\n{user_message}"
    )

    history_turns = build_native_history(llm_history)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history_turns)
    messages.append({"role": "user", "content": current_turn_content})

    resp = await client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=0.3,
    )
    raw = resp.choices[0].message.content or ""
    return strip_plan_block(raw)


async def check_reply(
    client,
    route: Dict[str, Any],
    checkpoint: CheckpointResult,
    decision: SupportDecision,
    draft_reply: str,
    llm_history: List[Dict[str, Any]],
    user_message: str,
) -> CheckResult:
    """Check the draft reply for safety and solution leaks; skip if the support level is low-risk."""
    if not should_run_safety_check(decision):
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
        "recent_history": compact_history(llm_history, limit=4),
        "route":          route,
    }

    raw_text, data, parse_ok = await call_json(
        client, system_prompt, json.dumps(payload), model=CHECK_MODEL,
    )

    if not parse_ok:
        logger.warning("check_reply: JSON parse failed — bypassing rewrite (was_skipped=True)")
        logger.warning("check_reply raw_text: %s", raw_text)
        return CheckResult(
            is_safe=True,
            leaks_solution=False,
            skipped_diagnosis=False,
            reason="check parse failed — safety check bypassed",
            was_skipped=True,
        )

    return CheckResult(
        is_safe=bool(data.get("is_safe", True)),
        leaks_solution=bool(data.get("leaks_solution", False)),
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
    """Rewrite a draft that failed the safety check using the fallback rewrite prompt."""
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
        "recent_history":   compact_history(llm_history, limit=4),
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


async def run_srl_chain(
    client,
    route: Dict[str, Any],
    llm_history: List[Dict[str, Any]],
    user_message: str,
) -> Dict[str, Any]:
    """Run the full SRL pipeline (checkpoint → generate → check → rewrite) and return all artifacts."""
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
        "reply":         final_reply,
        "draft_reply":   draft_reply,
        "diagnosis":     diagnosis.__dict__,
        "decision":      decision.__dict__,
        "check":         check.__dict__,
        "was_rewritten": final_reply != draft_reply,
    }
