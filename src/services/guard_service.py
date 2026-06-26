from __future__ import annotations

import logging
from typing import Optional

from utils.prompt_loader import load_prompt
from services.srl.srl_helpers import extract_json

logger = logging.getLogger(__name__)

GUARD_MODEL = "gpt-4o-mini"


class GuardService:
    """Detect direct-answer requests and return a guard hint to constrain generation."""

    def __init__(self, client):
        self.client = client
        self._classifier_system: Optional[str] = None

    async def get_hint(self, user_message: str) -> Optional[str]:
        """Return the guard prompt string if the message is a direct-answer request, else None."""
        should_guard = await self._classify(user_message)
        if not should_guard:
            return None

        logger.info("GuardService: direct-answer request detected — loading guard hint.")

        try:
            return load_prompt("constraints/direct_answer_guard.txt")
        except Exception as exc:
            logger.warning("GuardService: could not load guard prompt (%s) — guard will not fire.", exc)
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_classifier_system(self) -> str:
        if self._classifier_system is None:
            self._classifier_system = load_prompt("chains/guard.txt")
        return self._classifier_system

    async def _classify(self, user_message: str) -> bool:
        """Call the guard classifier LLM and return True if a guard should fire."""
        try:
            resp = await self.client.chat.completions.create(
                model=GUARD_MODEL,
                messages=[
                    {"role": "system", "content": self._get_classifier_system()},
                    {"role": "user", "content": f"STUDENT MESSAGE:\n{user_message}"},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = resp.choices[0].message.content or ""
            data = extract_json(raw)
            return bool(data.get("should_guard", False))
        except Exception as exc:
            logger.warning("GateService._classify failed (%s) — gate will not fire.", exc)
            return False