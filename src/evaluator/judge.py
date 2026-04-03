import json
from typing import Any, Dict

JUDGE_SYSTEM = """You are an impartial evaluator (AI-as-a-Judge).
Score strictly from the rubric and transcript evidence only.
Do not guess missing evidence. Return ONLY valid JSON."""


def build_judge_user_prompt(rubric: Dict[str, Any], transcript: str) -> str:
    return f"""
RUBRIC (JSON):
{json.dumps(rubric, ensure_ascii=False, indent=2)}

TRANSCRIPT:
{transcript}

SCORING PROCEDURE:
1) Extract 1-3 short verbatim evidence quotes for each criterion.
2) Compare evidence to rubric anchors and choose one best score (no averaging).
3) In rationale, state why score is not lower and not higher.
4) If evidence is mixed, use middle score only when both adjacent scores are explicitly ruled out.

ANTI-MIDDLE-SCORE RULE:
- Do not default to score 3.
- If evidence clearly matches 2 or 4, do not select 3.

OUTPUT JSON SCHEMA:
{{
  "rubric_name": string,
  "score_per_criterion": {{
    "<criterion_id>": {{
      "score": number,
      "rationale": "evidence: ... | why_not_lower: ... | why_not_higher: ...",
      "evidence_quotes": [string, ...]
    }},
    ...
  }},
  "overall_score": number,
  "fail_flags": [string, ...],
  "summary": [string, ...]
}}
""".strip()
