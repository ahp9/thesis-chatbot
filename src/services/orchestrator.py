from __future__ import annotations

from lib.contracts import CombinedControlResult, TurnResult
from lib.enums import Phase
from services.history_adapter import recent_control_state, recent_support_levels
from services.classify_service import ClassifyService
from services.generate_service import GenerateService
from services.policy.policy_engine import PolicyEngine
from services.router_service import RouterService
from services.safety_service import SafetyService
from services.telemetry import Telemetry


class Orchestrator:
    def __init__(self, client):
        self.router = RouterService(client)
        self.classify = ClassifyService(client)
        self.generator = GenerateService(client)
        self.safety = SafetyService(client)
        self.policy = PolicyEngine()
        self.telemetry = Telemetry()

    async def handle_turn(
        self,
        user_message: str,
        llm_history: list[dict],
        current_phase: Phase,
    ) -> TurnResult:
        route = await self.router.route(user_message, llm_history, current_phase)

        router_predicted_phase = route.phase

        # Pull the previous turn's frustration level so resolve_phase can raise
        # the confidence threshold before switching phase when the student was
        # highly frustrated. A frustrated student saying "I give up" reads like
        # re-orientation language but is usually still mid-task PERFORMANCE.
        previous_state = recent_control_state(llm_history)
        previous_frustration = previous_state.get("previous_frustration_level")

        resolved_phase = self.policy.resolve_phase(
            current_phase=current_phase,
            predicted_phase=router_predicted_phase,
            confidence=route.confidence,
            previous_frustration=previous_frustration,
        )

        route.phase = resolved_phase

        control, _ = await self.classify.classify(
            route.to_dict(),
            llm_history,
            user_message,
        )

        recent_levels = recent_support_levels(llm_history)

        policy_decision = self.policy.enforce_decision(
            checkpoint=control.checkpoint,
            decision=control.decision,
            recent_support_levels=recent_levels,
        )

        final_control = CombinedControlResult(
            checkpoint=control.checkpoint,
            decision=policy_decision,
        )

        draft_reply = await self.generator.generate(
            route,
            final_control,
            llm_history,
            user_message,
        )

        final_reply, safety, was_rewritten = await self.safety.enforce(
            route,
            final_control,
            draft_reply,
            llm_history,
            user_message,
        )

        return TurnResult(
            reply=final_reply,
            draft_reply=draft_reply,
            route=route,
            control=final_control,
            safety=safety,
            was_rewritten=was_rewritten,
            prefix="*(Self-Correction)*: " if was_rewritten else "",
        )