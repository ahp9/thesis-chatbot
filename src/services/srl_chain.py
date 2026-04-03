import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from services.history_adapter import last_assistant_reply, recent_control_state
from services.policy.policy_config import response_prompt_file_for
from services.prompt_loader import load_prompt

logger = logging.getLogger(__name__)

CONTROL_MODEL = "gpt-4.1-mini"     # diagnosis + support decision
GENERATION_MODEL = "gpt-4.1-mini"   # reply writing
CHECK_MODEL = "gpt-4o-mini"        # safety check
REWRITE_MODEL = "gpt-4o-mini"      # rewrite if needed

# Support levels where a safety/leak check is actually meaningful.
SAFETY_CHECK_LEVELS = {"PARTIAL", "EXPLAIN", "STRUCTURE", "EVALUATION"}


# ---------------------------------------------------------------------------
# Depth instruction table
#
# These instructions are injected directly into the generation prompt so the
# model sees an explicit behavioural contract, not just a label.
# The key insight: support_depth was previously computed and then ignored at
# generation time. This table closes that gap.
# ---------------------------------------------------------------------------

_DEPTH_INSTRUCTIONS: Dict[str, str] = {
    "SURFACE": (
        "DEPTH INSTRUCTION: Keep this response brief and orienting. "
        "One concept, one pointer. Do not go beyond what the student needs "
        "to take their very next step."
    ),
    "SURFACE_PLUS": (
        "DEPTH INSTRUCTION: Give a brief multi-part orientation. "
        "Name the relevant concept and provide one small grounding step or example. "
        "Do not expand into trade-offs or design considerations yet — the student "
        "needs to establish the basics before moving further."
    ),
    "SUBSTANTIVE": (
        "DEPTH INSTRUCTION: Engage with the student's actual decision or problem at a "
        "domain-appropriate level. Name specific options, trade-offs, or failure modes "
        "relevant to their specific case. Do not define terms they already used correctly. "
        "Do not ask what method or tool they plan to use if they already named one. "
        "The response must help them choose, evaluate, or act — not restate what they know."
    ),
    "SUBSTANTIVE_PLUS": (
        "DEPTH INSTRUCTION: Go beyond naming options — engage with the specific "
        "trade-offs, constraints, or design considerations the student is navigating. "
        "This student is past standard orientation. They need a response that "
        "addresses multiple considerations at once, compares alternatives in their "
        "specific context, or explains why the standard approach may or may not fit. "
        "Do not ask setup questions. Do not define terms they used correctly."
    ),
    "DEEP": (
        "DEPTH INSTRUCTION: Respond at a strategic or technical level. Assume full fluency. "
        "Skip procedural steps and basic orientation entirely. "
        "Raise a non-obvious consideration, edge case, design tension, or failure mode "
        "the student has not yet surfaced. Give one targeted insight that assumes "
        "the student already understands and has applied the standard approach. "
        "Go to where the standard approach breaks, becomes ambiguous, or requires "
        "architectural judgment."
    ),
}

# ---------------------------------------------------------------------------
# Affective state instruction table
#
# Injected alongside depth so the generator adjusts both content depth AND
# tone to the student's emotional state. Frustration detection lives entirely
# in the chain (checkpoint_and_decide), not in the router — this is the
# correct place because the chain has full structured context.
# ---------------------------------------------------------------------------

_AFFECTIVE_INSTRUCTIONS: Dict[str, str] = {
    "LOW": (
        "AFFECTIVE INSTRUCTION: The student appears calm and engaged. "
        "Maintain a direct, focused tone. Prioritise intellectual challenge over reassurance."
    ),
    "MEDIUM": (
        "AFFECTIVE INSTRUCTION: The student shows signs of uncertainty or mild frustration. "
        "Acknowledge the difficulty briefly before moving forward. "
        "Keep the next step clear and achievable. Avoid adding more questions than necessary."
    ),
    "HIGH": (
        "AFFECTIVE INSTRUCTION: The student is clearly frustrated or stuck. "
        "Do not add clarifying questions. Do not ask them to try again without guidance. "
        "Provide enough concrete direction that they can take one clear action immediately. "
        "A short acknowledgement of the difficulty is appropriate, but move quickly to the help."
    ),
}


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
    parse_ok: bool = True          # False when the model response could not be parsed


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
    "identity": "base/srl_model_v3.txt",
    "phase_forethought": "phases/forethought_core_v2.txt",
    "phase_performance": "phases/performance_core_v1.txt",
    "phase_reflection": "phases/reflection_core.txt",
    "diagnose": "chains/diagnose_student.txt",
    "decide_support": "chains/choose_support_level.txt",
    "check_reply": "chains/check_solution_leak_v3.txt",
    "rewrite_reply": "chains/fallback_rewrite_v3.txt",
    "checkpoint_and_decide": "chains/student_state_v5.txt",
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

    Returns an empty dict only if all attempts fail, and logs the raw
    response so you can see exactly what the model returned.
    """
    if not raw or not raw.strip():
        logger.warning("_extract_json: model returned empty response")
        return {}

    # Attempt 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Attempt 2: strip ```json ... ``` or ``` ... ``` fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Attempt 3: find the first { ... } block in the response
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # All attempts failed — log so it is visible in your terminal
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

def _phase_prompt_file(phase: str | None) -> str:
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
        # Ensure we don't lose the markers that signal expertise
        if content:
            lines.append(f"{role}: {content[:500]}") # Cap length for speed
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

    # OpenAI requires the list to start with a user turn
    while clean and clean[0]["role"] == "assistant":
        clean.pop(0)

    return clean

def _build_checkpoint_payload(
    route: Dict[str, Any],
    llm_history: List[Dict[str, Any]],
    user_message: str,
) -> str:
    
    recent_control = recent_control_state(llm_history)
    parts = [
        f"CURRENT_PHASE:\n{route.get('phase', 'UNKNOWN')}",
        f"PREVIOUS_CONTROL:\n{json.dumps(recent_control, indent=2)}",
        f"RECENT_HISTORY:\n{_compact_history(llm_history)}",
        f"CURRENT_USER_MESSAGE:\n{user_message}",
    ]
    return "\n\n".join(parts)


def _should_run_safety_check(decision: SupportDecision) -> bool:
    """
    Run the check when the support level can plausibly produce a leaked
    solution or runnable code.  Skip for structurally safe levels.
    """
    if decision.can_show_code:
        return True
    if decision.support_level in SAFETY_CHECK_LEVELS:
        return True
    return False


def _build_depth_and_affective_header(
    checkpoint: CheckpointResult,
    decision: SupportDecision,
) -> str:
    """
    Build the depth + affective instruction block that is prepended to the
    generation system prompt.

    This is the single place where support_depth and frustration_level are
    translated from labels into explicit behavioural instructions the model
    can act on. Without this, those values are computed correctly but then
    ignored at generation time.
    """
    depth_key = (decision.support_depth or "SUBSTANTIVE").upper()
    affective_key = (checkpoint.frustration_level or "LOW").upper()

    depth_instr = _DEPTH_INSTRUCTIONS.get(depth_key, _DEPTH_INSTRUCTIONS["SUBSTANTIVE"])
    affective_instr = _AFFECTIVE_INSTRUCTIONS.get(affective_key, _AFFECTIVE_INSTRUCTIONS["LOW"])

    return f"{depth_instr}\n\n{affective_instr}"


# ---------------------------------------------------------------------------
# Fallback values — used ONLY when parsing genuinely fails.
# All rationale fields say PARSE_FAILED so they are visible in logs.
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
) -> tuple[CheckpointResult, SupportDecision, dict]:
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

async def generate_full_reply(
    client,
    route: Dict[str, Any],
    checkpoint: CheckpointResult,
    decision: SupportDecision,
    llm_history: List[Dict[str, Any]],
    user_message: str,
) -> str:
    """
    Generate the tutor reply using native multi-turn history.

    The system prompt is built in layers:
      1. SRL identity
      2. Phase prompt (forethought / performance / reflection)
      3. Support-level response prompt
      4. Depth + affective instruction block  ← new: translates support_depth
         and frustration_level into explicit behavioural contracts the model
         can act on at generation time
      5. File handler (conditional)

    History is passed as real alternating user/assistant turns rather than
    a serialised string, which reduces token use and improves coherence.
    """
    depth_and_affective = _build_depth_and_affective_header(checkpoint, decision)

    prompt_parts = [
        load_prompt(BASE_PROMPT_FILES["identity"]),
        load_prompt(_phase_prompt_file(route.get("phase"))),
        load_prompt(decision.response_prompt_file),
        depth_and_affective,
    ]
    if _has_file_content(user_message):
        prompt_parts.append(load_prompt(BASE_PROMPT_FILES["file_handler"]))

    system_prompt = "\n\n".join(prompt_parts)

    control_header = json.dumps(
        {
            "route": route,
            "checkpoint": checkpoint.__dict__,
            "decision": decision.__dict__,
        },
        indent=2,
    )

    # Inject the last assistant reply so the generator knows what it just said.
    # Without this, GPT-4o-mini regenerates the same structural breakdown every
    # turn because it only sees the decision label (e.g. "STRUCTURE") and not
    # the actual content it produced for that label last time.
    previous_reply = last_assistant_reply(llm_history)

    current_turn_content = (
        f"CONTROL_STATE:\n{control_header}\n\n"
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
    return resp.choices[0].message.content or ""


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
        # Fail safe: if we can't read the check result, assume the reply needs rewriting
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

    # Pass both what the draft said AND what the previous turn said.
    # The rewrite needs to know both so it doesn't clone either of them.
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