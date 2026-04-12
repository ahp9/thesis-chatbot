from __future__ import annotations

import logging
from typing import Optional

from services.prompt_loader import load_prompt
from services.srl_chain import _extract_json

logger = logging.getLogger(__name__)

GUARD_MODEL = "gpt-4o-mini"

_GUARD_CLASSIFIER_SYSTEM = (
    "You classify student messages. Return JSON only — no prose, no fences.\n\n"
    "Decide: is the student's opening message asking for a direct answer, "
    "calculation, result, or a conclusion derived from data — with no prior attempt shown?\n\n"
    "Return exactly one of:\n"
    '  {"should_guard": true}\n'
    '  {"should_guard": false}\n\n'
    "Mark should_guard=true when the message:\n"
    "  - asks the tutor to compute, calculate, or run something directly\n"
    "  - requests a final answer, result, or interpretation of a relationship (e.g., correlation, trends, or differences)\n"
    "  - asks the tutor to determine if a specific hypothesis is true based on the data\n"
    "  - would be fully resolved by a number, table, or a 'yes/no' conclusion about the data\n\n"
    "Mark should_guard=false when the message:\n"
    "  - describes something the student has already tried\n"
    "  - asks for a definition of a concept (e.g., 'What is correlation?')\n"
    "  - requests guidance on the steps to approach a task\n"
    "  - shares code or partial work and asks for feedback\n"
    "  - uploads a file and asks what it contains (metadata only)"
)


class GuardService:
    def __init__(self, client):
        self.client = client

    async def get_hint(self, user_message: str) -> Optional[str]:
        should_guard = await self._classify(user_message)
        if not should_guard:
            return None

        logger.info("GuardService: direct-answer request detected — loading guard hint.")

        try:
            return load_prompt("chains/direct_answer_guard.txt")
        except Exception as exc:
            logger.warning("GuardService: could not load guard prompt (%s) — guard will not fire.", exc)
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _classify(self, user_message: str) -> bool:
        try:
            resp = await self.client.chat.completions.create(
                model=GUARD_MODEL,
                messages=[
                    {"role": "system", "content": _GUARD_CLASSIFIER_SYSTEM},
                    {"role": "user", "content": f"STUDENT MESSAGE:\n{user_message}"},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = resp.choices[0].message.content or ""
            data = _extract_json(raw)
            return bool(data.get("should_guard", False))
        except Exception as exc:
            logger.warning("GateService._classify failed (%s) — gate will not fire.", exc)
            return False


