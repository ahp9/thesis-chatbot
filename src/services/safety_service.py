from __future__ import annotations

from lib.contracts import CombinedControlResult, RouteResult, SafetyResult
from services.srl_chain import (
    CheckpointResult,
    SupportDecision,
    check_reply,
    rewrite_reply,
)


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
        
        checkpoint = CheckpointResult(
            request_kind=control.checkpoint.request_kind.value,
            task_stage=control.checkpoint.task_stage.value,
            progress_state=control.checkpoint.progress_state.value,
            has_attempt=control.checkpoint.has_attempt,
            context_gap=control.checkpoint.context_gap.value,
            expertise_level=control.checkpoint.expertise_level.value,
            frustration_level=control.checkpoint.frustration_level.value,
            srl_focus=control.checkpoint.srl_focus.value,
            implementation_allowed=control.checkpoint.implementation_allowed,
            confidence=control.checkpoint.confidence,
            rationale=control.checkpoint.rationale,
            parse_ok=control.checkpoint.parse_ok,
        )

        decision = SupportDecision(
            support_level=control.decision.support_level.value,
            response_prompt_file=control.decision.response_prompt_file,
            can_show_code=control.decision.can_show_code,
            must_end_with_question=control.decision.must_end_with_question,
            should_request_attempt=control.decision.should_request_attempt,
            confidence=control.decision.confidence,
            rationale=control.decision.rationale,
            support_depth=control.decision.support_depth.value,
            parse_ok=control.decision.parse_ok,
        )
        check_raw = await check_reply(
            self.client,
            route.to_dict(),
            checkpoint,
            decision,
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
                checkpoint,
                decision,
                draft_reply,
                check_raw,
                llm_history,
                user_message,
            )
            return rewritten, check, True

        return draft_reply, check, False