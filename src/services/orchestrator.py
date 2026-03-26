from __future__ import annotations

from lib.contracts import CombinedControlResult, TurnResult
from lib.enums import Phase, SupportLevel
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

    def _recent_support_levels(self, llm_history: list[dict]) -> list[SupportLevel]:
        levels: list[SupportLevel] = []
        for item in llm_history:
            if item.get("role") != "assistant":
                continue

            decision = item.get("decision")
            if not isinstance(decision, dict):
                continue

            raw = decision.get("support_level")
            try:
                levels.append(SupportLevel(raw))
            except Exception:
                continue

        return levels[-3:]

    async def handle_turn(
        self,
        user_message: str,
        llm_history: list[dict],
        current_phase: Phase,
    ) -> TurnResult:
        route = await self.router.route(user_message, llm_history, current_phase)
        #self.telemetry.event("route.completed", route.to_dict())

        router_predicted_phase = route.phase
        resolved_phase = self.policy.resolve_phase(
            current_phase=current_phase,
            predicted_phase=router_predicted_phase,
            confidence=route.confidence,
        )

        # self.telemetry.event(
        #     "policy.phase_compare",
        #     {
        #         "current_phase": current_phase.value,
        #         "router_predicted_phase": router_predicted_phase.value,
        #         "router_confidence": route.confidence,
        #         "final_phase": resolved_phase.value,
        #         "phase_was_overridden": router_predicted_phase != resolved_phase,
        #     },
        # )

        route.phase = resolved_phase

        control, classify_debug = await self.classify.classify(
            route.to_dict(),
            llm_history,
            user_message,
        )

        # self.telemetry.event("control.completed", control.to_dict())
        # self.telemetry.event("control.debug", classify_debug)

        recent_support_levels = self._recent_support_levels(llm_history)

        policy_decision = control.decision

        # self.telemetry.event(
        #     "policy.decision_compare",
        #     {
        #         "llm_decision": control.decision.to_dict(),
        #         "policy_decision": policy_decision.to_dict(),
        #         "recent_support_levels": [lvl.value for lvl in recent_support_levels],
        #         "decision_was_overridden": control.decision.to_dict() != policy_decision.to_dict(),
        #     },
        # )

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

        # self.telemetry.event(
        #     "generation.completed",
        #     {
        #         "reply_preview": draft_reply[:300],
        #         "phase": route.phase.value,
        #         "support_level": final_control.decision.support_level.value,
        #     },
        # )

        final_reply, safety, was_rewritten = await self.safety.enforce(
            route,
            final_control,
            draft_reply,
            llm_history,
            user_message,
        )

        # self.telemetry.event("safety.completed", safety.to_dict())

        return TurnResult(
            reply=final_reply,
            draft_reply=draft_reply,
            route=route,
            control=final_control,
            safety=safety,
            was_rewritten=was_rewritten,
            prefix="*(Self-Correction)*: " if was_rewritten else "",
        )