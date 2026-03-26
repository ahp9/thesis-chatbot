from __future__ import annotations

from lib.contracts import CombinedControlResult, RouteResult
from services.srl_chain import generate_full_reply


class GenerateService:
    def __init__(self, client):
        self.client = client

    async def generate(
        self,
        route: RouteResult,
        control: CombinedControlResult,
        llm_history: list[dict],
        user_message: str,
    ) -> str:
        return await generate_full_reply(
            self.client,
            route.to_dict(),
            control.checkpoint,
            control.decision,
            llm_history,
            user_message,
        )