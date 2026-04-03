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

    sections: list[str] = []

    sections.append(
        "SECTION: TASK\n"
        f"rubric_type: {rubric_type}\n"
        "Evaluate each criterion independently. Use only observable evidence."
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
        "1) Evidence extraction: for each criterion, list 1-3 short verbatim quotes.\n"
        "2) Fit check: compare evidence against score anchors. Pick the SINGLE best-matching score (no averaging).\n"
        "3) Boundary check: explain why the score is not one level lower and not one level higher.\n"
        "4) If criterion is not applicable, set applicable=false and score=null."
    )

    sections.append(
        "ANTI-MIDDLE-SCORE SAFEGUARDS:\n"
        "- Score 3 is allowed only when evidence shows true mixed/partial performance.\n"
        "- If evidence aligns clearly with score 2 or 4, do NOT use score 3.\n"
        "- If you output score 3, rationale must explicitly state both: why not 2 and why not 4.\n"
        "- Do not infer intent. If evidence is missing, lower confidence in rationale, not score inflation."
    )

    sections.append(
        "OUTPUT CONTRACT:\n"
        "- Return ONLY valid JSON (no markdown).\n"
        "- Keep summary to max 5 bullets.\n"
        "- overall_score = arithmetic mean of applicable scores only, rounded to 1 decimal.\n"
        "- rationale format (required): 'evidence: ... | why_not_lower: ... | why_not_higher: ...'\n"
        f"Required schema:\n{_pretty_json(OUTPUT_SCHEMA)}"
    )

    return "\n\n".join(sections)