import json
from typing import Any, Dict

JUDGE_SYSTEM = """You are an impartial evaluator (AI-as-a-Judge).
Your job is to score the assistant's behavior against the provided rubric.
You must be strict, consistent, and cite evidence from the transcript.

Return ONLY valid JSON matching the requested schema.
Do not provide advice to the student. Do not rewrite the assistant response.
"""

def build_judge_user_prompt(rubric: Dict[str, Any], transcript: str) -> str:
    return f"""
RUBRIC (YAML converted to JSON):
{json.dumps(rubric, ensure_ascii=False, indent=2)}

TRANSCRIPT:
{transcript}

INSTRUCTIONS:
- Score each criterion on the rubric scale.
- Provide 1–3 short evidence quotes (verbatim excerpts) from the assistant that justify each score.
- If any fail_flags apply, include them.
- Provide an overall_score (average, rounded to 1 decimal).
- Provide a short summary (max 5 bullet points).

OUTPUT JSON SCHEMA:
{{
  "rubric_name": string,
  "score_per_criterion": {{
    "<criterion_id>": {{
      "score": number,
      "rationale": string,
      "evidence_quotes": [string, ...]
    }},
    ...
  }},
  "overall_score": number,
  "fail_flags": [string, ...],
  "summary": [string, ...]
}}
""".strip()