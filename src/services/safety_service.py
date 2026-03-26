from __future__ import annotations

from lib.contracts import CombinedControlResult, RouteResult, SafetyResult
from services.srl_chain import check_reply, rewrite_reply


class SafetyService:
    def __init__(self, client):
        self.client = client

    async def enforce(
        self,
        route: RouteResult,
        control: CombinedControlResult,
        draft_reply: str,
        llm_history: list[dict],
        user_message: str,
    ) -> tuple[str, SafetyResult, bool]:
        check_raw = await check_reply(
            self.client,
            route.to_dict(),
            control.checkpoint,
            control.decision,
            draft_reply,
            llm_history,
            user_message,
        )

        check = SafetyResult(
            is_safe=bool(check_raw.is_safe),
            leaks_solution=bool(check_raw.leaks_solution),
            skipped_diagnosis=bool(check_raw.skipped_diagnosis),
            reason=str(check_raw.reason),
            was_skipped=bool(check_raw.was_skipped),
        )

        if not check.is_safe or check.leaks_solution:
            rewritten = await rewrite_reply(
                self.client,
                route.to_dict(),
                control.checkpoint,
                control.decision,
                draft_reply,
                check_raw,
                llm_history,
                user_message,
            )
            return rewritten, check, True

        return draft_reply, check, False