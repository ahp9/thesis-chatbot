import json
from typing import Any, Dict, Optional


JUDGE_SYSTEM = """
    ROLE: You are an impartial evaluator (AI-as-a-Judge).
    
    PURPOSE: Your task is to evaluate the assistant strictly against the provided rubric.
    
    CORE REQUIREMENTS:
    - Be precise, evidence-based, and consistent.
    - Judge only what is supported by the transcript and any chain internals provided.
    - Do not invent evidence.
    - Do not give advice, coaching, or rewrites.
    - If evidence is missing, say so in the rationale rather than guessing.
    - Return ONLY valid JSON matching the required schema.
    """

RUBRIC_TYPE_DESCRIPTIONS = {
    "router": (
        "You are evaluating the ROUTER decision only. "
        "Focus on whether the phase classification is correct, well-supported, "
        "and appropriately confident. The transcript is supporting context."
    ),
    "chain": (
        "You are evaluating the INTERNAL CHAIN DECISIONS only. "
        "Focus on diagnosis quality, support-level selection, checks, and whether "
        "the internal reasoning objects align with the rubric. The final response "
        "is supporting context, not the primary target."
    ),
    "response": (
        "You are evaluating the FINAL RESPONSE shown to the student. "
        "Focus primarily on the response text in the transcript. "
        "Use chain internals only to check fidelity, consistency, and whether the "
        "final reply matches the intended behavior."
    ),
}

RUBRIC_INPUT_GUIDANCE = {
    "router": (
        "Prioritize: route object.\n"
        "Secondary context: transcript.\n"
        "Ignore response style unless the rubric explicitly requires it."
    ),
    "chain": (
        "Prioritize: diagnosis, decision, check, and rewrite status.\n"
        "Secondary context: transcript.\n"
        "Do not over-penalize final phrasing unless chain fidelity is part of the rubric."
    ),
    "response": (
        "Prioritize: transcript and final reply.\n"
        "Secondary context: route/diagnosis/decision/check/draft reply for fidelity checking."
    ),
}

OUTPUT_SCHEMA = {
    "rubric_name": "string",
    "rubric_type": "string",
    "score_per_criterion": {
        "<criterion_id>": {
            "applicable": "boolean",
            "score": "number | null",
            "rationale": "string",
            "evidence_quotes": ["string"]
        }
    },
    "overall_score": "number",
    "fail_flags": ["string"],
    "summary": ["string"]
}


def _pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def _select_chain_view(
    rubric_type: str,
    chain_internals: Dict[str, Any],
) -> Dict[str, Any]:
    if rubric_type == "router":
        return {
            "route": chain_internals.get("route")
        }

    if rubric_type == "chain":
        return {
            "route": chain_internals.get("route"),
            "diagnosis": chain_internals.get("diagnosis"),
            "decision": chain_internals.get("decision"),
            "check": chain_internals.get("check"),
            "was_rewritten": chain_internals.get("was_rewritten", False),
        }

    return {
        "route": chain_internals.get("route"),
        "diagnosis": chain_internals.get("diagnosis"),
        "decision": chain_internals.get("decision"),
        "check": chain_internals.get("check"),
        "draft_reply": chain_internals.get("draft_reply"),
        "final_reply": chain_internals.get("final_reply"),
        "was_rewritten": chain_internals.get("was_rewritten", False),
    }


def _build_evidence_rules() -> str:
    return """EVIDENCE RULES:
                - Base every score on explicit evidence from the transcript and/or provided chain internals.
                - Use 1-3 short verbatim evidence quotes for each criterion.
                - Prefer the shortest quote that proves the point.
                - Do not fabricate quotes.
                - If evidence is insufficient, state that explicitly in the rationale rather than guessing.
            """


def _build_scoring_rules() -> str:
    return """SCORING RULES:
                - Score each criterion exactly on the scale defined in the rubric.
                - Apply the rubric strictly.
                - Respect applicability notes in the rubric.
                - If a criterion is not applicable, set:
                  - applicable = false
                  - score = null
                  - rationale = brief explanation of why it is not applicable
                - If a criterion is applicable, set:
                  - applicable = true
                  - score = numeric rubric score
                - Do not inflate scores for partially met criteria.
                - If fail_flags apply, include them by exact name.
                - Compute overall_score as the arithmetic mean of APPLICABLE criterion scores only, rounded to 1 decimal.
                - Exclude any criterion with score = null from the mean.
                - Keep summary to at most 5 bullets.
            """


def _build_output_contract() -> str:
    return (
        "OUTPUT CONTRACT:\n"
        "Return ONLY valid JSON.\n"
        "Do not wrap the JSON in markdown fences.\n"
        "Do not include commentary before or after the JSON.\n"
        f"Required schema:\n{_pretty_json(OUTPUT_SCHEMA)}"
    )


def build_judge_user_prompt(
    rubric: Dict[str, Any],
    transcript: str,
    chain_internals: Optional[Dict[str, Any]] = None,
) -> str:
    rubric_type = str(rubric.get("rubric_type", "response")).strip().lower()
    if rubric_type not in RUBRIC_TYPE_DESCRIPTIONS:
        rubric_type = "response"

    type_instruction = RUBRIC_TYPE_DESCRIPTIONS[rubric_type]
    input_guidance = RUBRIC_INPUT_GUIDANCE[rubric_type]

    assistant_turn_count = transcript.count("ASSISTANT:")
    user_turn_count = transcript.count("USER:")

    sections: list[str] = []

    sections.append(
        "SECTION: EVALUATION FOCUS\n"
        f"rubric_type: {rubric_type}\n"
        f"{type_instruction}"
    )

    sections.append(
        "SECTION: INPUT PRIORITIZATION\n"
        f"{input_guidance}"
    )

    sections.append(
        "SECTION: RUBRIC\n"
        "The following rubric was originally written in YAML and converted to JSON:\n"
        f"{_pretty_json(rubric)}"
    )

    if chain_internals:
        relevant_chain = _select_chain_view(rubric_type, chain_internals)
        sections.append(
            "SECTION: CHAIN INTERNALS\n"
            "Use only the fields below when they are relevant to the rubric:\n"
            f"{_pretty_json(relevant_chain)}"
        )

    sections.append(
        "SECTION: CONVERSATION METADATA\n"
        f"user_turn_count: {user_turn_count}\n"
        f"assistant_turn_count: {assistant_turn_count}\n"
        "For criteria requiring prior tutor turns, at least one assistant turn must exist before the evaluated response."
    )

    sections.append(
        "SECTION: TRANSCRIPT\n"
        "This is the ground-truth interaction to evaluate:\n"
        f"{transcript}"
    )

    sections.append(_build_evidence_rules())
    sections.append(_build_scoring_rules())

    sections.append(
        "GUARDRAILS:\n"
        "- Do not infer hidden intent unless supported by evidence.\n"
        "- Do not reward style when the rubric targets routing or chain logic.\n"
        "- Do not penalize missing chain fields if they were not provided.\n"
        "- Do not use external knowledge.\n"
        "- If transcript and chain internals conflict, note the conflict explicitly.\n"
        "- If a criterion says it applies only when prior tutor turns exist, mark it not applicable when the transcript does not contain a prior assistant turn before the evaluated response."
    )

    sections.append(_build_output_contract())

    return "\n\n".join(sections)