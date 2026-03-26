from __future__ import annotations

from lib.contracts import Checkpoint, CombinedControlResult, Decision
from lib.enums import (
    ContextGap,
    ExpertiseLevel,
    FrustrationLevel,
    ProgressState,
    RequestKind,
    SRLFocus,
    SupportDepth,
    SupportLevel,
    TaskStage,
)
from services.policy.policy_config import RESPONSE_PROMPT_FILES
from services.srl_chain import checkpoint_and_decide


def _safe_enum(enum_cls, value, default):
    try:
        return enum_cls(value)
    except Exception:
        return default


class ClassifyService:
    def __init__(self, client):
        self.client = client

    async def classify(
        self,
        route: dict,
        llm_history: list[dict],
        user_message: str,
    ) -> tuple[CombinedControlResult, dict]:
        checkpoint_raw, decision_raw, debug = await checkpoint_and_decide(
            self.client,
            route,
            llm_history,
            user_message,
        )

        checkpoint = Checkpoint(
            request_kind=_safe_enum(
                RequestKind,
                checkpoint_raw.request_kind,
                RequestKind.PRODUCT,
            ),
            task_stage=_safe_enum(
                TaskStage,
                checkpoint_raw.task_stage,
                TaskStage.WORKING,
            ),
            progress_state=_safe_enum(
                ProgressState,
                checkpoint_raw.progress_state,
                ProgressState.MOVING,
            ),
            has_attempt=bool(checkpoint_raw.has_attempt),
            context_gap=_safe_enum(
                ContextGap,
                checkpoint_raw.context_gap,
                ContextGap.SMALL,
            ),
            expertise_level=_safe_enum(
                ExpertiseLevel,
                checkpoint_raw.expertise_level,
                ExpertiseLevel.NOVICE,
            ),
            frustration_level=_safe_enum(
                FrustrationLevel,
                checkpoint_raw.frustration_level,
                FrustrationLevel.LOW,
            ),
            srl_focus=_safe_enum(
                SRLFocus,
                checkpoint_raw.srl_focus,
                SRLFocus.STRATEGY,
            ),
            implementation_allowed=bool(checkpoint_raw.implementation_allowed),
            confidence=float(checkpoint_raw.confidence or 0.0),
            rationale=list(checkpoint_raw.rationale or []),
            parse_ok=bool(getattr(checkpoint_raw, "parse_ok", True)),
        )

        support_level = _safe_enum(
            SupportLevel,
            decision_raw.support_level,
            SupportLevel.QUESTION,
        )

        decision = Decision(
            support_level=support_level,
            response_prompt_file=RESPONSE_PROMPT_FILES[support_level],
            support_depth=_safe_enum(
                SupportDepth,
                getattr(decision_raw, "support_depth", "SUBSTANTIVE"),
                SupportDepth.SUBSTANTIVE,
            ),
            can_show_code=bool(decision_raw.can_show_code),
            must_end_with_question=bool(decision_raw.must_end_with_question),
            should_request_attempt=bool(decision_raw.should_request_attempt),
            confidence=float(decision_raw.confidence or 0.0),
            rationale=list(decision_raw.rationale or []),
            parse_ok=bool(getattr(decision_raw, "parse_ok", True)),
        )

        classify_debug = {
            "raw_checkpoint_obj": checkpoint_raw.__dict__,
            "raw_decision_obj": decision_raw.__dict__,
            "normalized_checkpoint": checkpoint.to_dict(),
            "normalized_decision": decision.to_dict(),
        }
        classify_debug.update(debug)

        return CombinedControlResult(
            checkpoint=checkpoint,
            decision=decision,
        ), classify_debug