from __future__ import annotations

import logging
from typing import Optional

from services.prompt_loader import load_prompt
from services.srl_chain import _extract_json

logger = logging.getLogger(__name__)

GATE_MODEL = "gpt-4o-mini"

_GATE_CLASSIFIER_SYSTEM = (
    "You classify student messages. Return JSON only — no prose, no fences.\n\n"
    "Decide: is the student's opening message asking for a direct answer, "
    "calculation, result, or complete solution — with no prior attempt shown?\n\n"
    "Return exactly one of:\n"
    '  {"should_gate": true}\n'
    '  {"should_gate": false}\n\n'
    "Mark should_gate=true when the message:\n"
    "  - asks the tutor to compute, calculate, or run something directly\n"
    "  - requests a final answer or result with no described attempt\n"
    "  - would be fully resolved by a number, table, or code block\n\n"
    "Mark should_gate=false when the message:\n"
    "  - describes something the student has already tried\n"
    "  - asks a conceptual or setup question\n"
    "  - requests guidance on how to approach a task\n"
    "  - shares code or partial work and asks for feedback\n"
    "  - uploads a file and asks what it contains"
)


class GateService:
    def __init__(self, client):
        self.client = client

    async def get_hint(self, user_message: str) -> Optional[str]:
        """
        Check whether the gate should fire for this first-turn message.

        Returns the loaded gate prompt string if the student is asking for
        a direct answer, else None. The caller (orchestrator) passes this
        string into generate() where it is prepended to the system prompt.
        """
        should_gate = await self._classify(user_message)
        if not should_gate:
            return None

        logger.info("GateService: direct-answer request detected — loading gate hint.")

        try:
            return load_prompt("chains/first_turn.txt")
        except Exception as exc:
            logger.warning("GateService: could not load gate prompt (%s) — gate will not fire.", exc)
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _classify(self, user_message: str) -> bool:
        """
        Binary classification: should the gate fire?
        Fails open (returns False) on any error so the normal chain always
        runs if something goes wrong.
        """
        try:
            resp = await self.client.chat.completions.create(
                model=GATE_MODEL,
                messages=[
                    {"role": "system", "content": _GATE_CLASSIFIER_SYSTEM},
                    {"role": "user", "content": f"STUDENT MESSAGE:\n{user_message}"},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = resp.choices[0].message.content or ""
            data = _extract_json(raw)
            return bool(data.get("should_gate", False))
        except Exception as exc:
            logger.warning("GateService._classify failed (%s) — gate will not fire.", exc)
            return False


