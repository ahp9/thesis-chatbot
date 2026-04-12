import json
from typing import Any, Dict, Optional

JUDGE_SYSTEM = """
ROLE: You are an impartial evaluator (AI-as-a-Judge).
Evaluate strictly using only transcript and provided chain fields.
Do not guess missing evidence. Do not coach or rewrite.
Return ONLY valid JSON.
""".strip()

OUTPUT_SCHEMA = {
    "rubric_name": "string",
    "rubric_type": "string",
    "score_per_criterion": {
        "<criterion_id>": {
            "applicable": "boolean",
            "score": "number | null",
            "rationale": "string",
            "evidence_quotes": ["string"],
            "turn_evidence": {"<turn_number>": "string | null"},
        }
    },
    "overall_score": "number",
    "fail_flags": ["string"],
    "summary": ["string"],
}


def _pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def _select_chain_view(rubric_type: str, chain_internals: Dict[str, Any]) -> Dict[str, Any]:
    snapshots = chain_internals.get("turn_snapshots")

    if not snapshots:
        return {
            "route": chain_internals.get("route"),
            "diagnosis": chain_internals.get("diagnosis"),
            "decision": chain_internals.get("decision"),
            "check": chain_internals.get("check"),
            "final_reply": chain_internals.get("final_reply"),
        }

    def _trim_snapshot(s: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "turn": s.get("turn"),
            "route": s.get("route"),
            "diagnosis": s.get("diagnosis"),
            "decision": s.get("decision"),
            "check": s.get("check"),
            "final_reply": s.get("final_reply"),
        }

    return {
        "total_turns": chain_internals.get("total_turns", len(snapshots)),
        "final_phase": chain_internals.get("final_phase"),
        "turn_snapshots": [_trim_snapshot(s) for s in snapshots],
    }


def build_judge_user_prompt(
    rubric: Dict[str, Any],
    transcript: str,
    chain_internals: Optional[Dict[str, Any]] = None,
) -> str:
    rubric_type = str(rubric.get("rubric_type", "response")).strip().lower()
    criteria = rubric.get("criteria", {}) or {}
    criterion_ids = list(criteria.keys())

    sections: list[str] = []

    sections.append(
        "SECTION: TASK\n"
        f"rubric_type: {rubric_type}\n"
        "Evaluate each criterion independently. Use only observable evidence."
    )

    sections.append(
        "SECTION: REQUIRED CRITERIA\n"
        "You MUST return one entry in score_per_criterion for EVERY criterion id below.\n"
        "Do not omit any criterion.\n"
        f"{_pretty_json(criterion_ids)}"
    )

    sections.append(
        "SECTION: RUBRIC\n"
        "Apply this rubric exactly as written:\n"
        f"{_pretty_json(rubric)}"
    )

    if chain_internals:
        sections.append(
            "SECTION: CHAIN CONTEXT\n"
            "Use only these fields when relevant:\n"
            f"{_pretty_json(_select_chain_view(rubric_type, chain_internals))}"
        )

    sections.append(
        "SECTION: TRANSCRIPT\n"
        "Ground-truth interaction:\n"
        f"{transcript}"
    )

    sections.append(
        "SCORING PROCEDURE (MANDATORY):\n"
        "1) Evidence extraction: for EACH criterion, list 1-3 short verbatim quotes.\n"
        "2) Fill turn_evidence for every assistant turn from 1 to total_turns.\n"
        "3) Fit check: compare evidence against score anchors. Pick the SINGLE best-matching score (no averaging).\n"
        "4) Boundary check: explain why the score is not one level lower and not one level higher.\n"
        "5) If criterion is not applicable, set applicable=false and score=null."
    )

    sections.append(
        "ANTI-DEFAULT-3 SAFEGUARDS:\n"
        "- Do not use score 3 as a safe middle.\n"
        "- Score 3 is allowed only when the rubric's middle anchor is the best fit.\n"
        "- If you assign score 3, rationale must explicitly explain why not the lower adjacent score and why not the higher adjacent score.\n"
        "- If evidence clearly matches a lower or higher anchor, use that score."
    )
    
    sections.append(
        "NUMERIC SCORE RULES:\n"
        "- If applicable=true, score MUST be a numeric value from the rubric scale.\n"
        "- If applicable=false, score MUST be null.\n"
        "- It is invalid to return applicable=true with score=null.\n"
        "- Before finishing, verify every required criterion has a numeric score or is explicitly not applicable."
    )
    
    sections.append(
        "TURN_EVIDENCE RULES:\n"
        "- For every criterion, turn_evidence must contain one key for every assistant turn number from 1 to total_turns.\n"
        "- If that turn contains relevant evidence for the criterion, set the value to 'relevant'.\n"
        "- If that turn is not relevant, set the value to null.\n"
        "- Do not omit any turn key.\n"
        "- Do not leave turn_evidence empty."
    )

    sections.append(
        "OUTPUT RULES:\n"
        "- Return ONLY valid JSON.\n"
        "- score_per_criterion must contain EVERY required criterion id exactly once.\n"
        "- For every criterion object include: applicable, score, rationale, evidence_quotes, turn_evidence.\n"
        "- rationale format: 'evidence: ... | why_not_lower: ... | why_not_higher: ...'\n"
        "- overall_score should be the arithmetic mean of applicable scores only, rounded to 1 decimal.\n"
        "- Keep summary to max 3 bullets.\n"
        f"Required schema:\n{_pretty_json(OUTPUT_SCHEMA)}"
    )

    return "\n\n".join(sections)