from __future__ import annotations

from dataclasses import replace
from typing import Set

from lib.contracts import Checkpoint, Decision
from lib.enums import (
    ContextGap,
    ExpertiseLevel,
    FrustrationLevel,
    ProgressState,
    SRLFocus,
    SupportDepth,
    SupportLevel,
)
from services.policy.policy_config import (
    CODE_ALLOWED_LEVELS,
    QUESTION_REQUIRED_LEVELS,
    SWITCH_THRESHOLD,
    TRANSITIONS,
    response_prompt_file_for,
)

# Hard depth floors per expertise level.
# ADVANCED gets SUBSTANTIVE_PLUS as the minimum — DEEP is the target for
# primary questions but SUBSTANTIVE_PLUS is acceptable for contained sub-questions.
_DEPTH_FLOOR: dict[ExpertiseLevel, SupportDepth] = {
    ExpertiseLevel.NOVICE:        SupportDepth.SURFACE,
    ExpertiseLevel.INTERMEDIATE:  SupportDepth.SUBSTANTIVE,
    ExpertiseLevel.ADVANCED:      SupportDepth.SUBSTANTIVE_PLUS,
}

_DEPTH_ORDER = {
    SupportDepth.SURFACE: 1,
    SupportDepth.SURFACE_PLUS: 2,
    SupportDepth.SUBSTANTIVE: 3,
    SupportDepth.SUBSTANTIVE_PLUS: 4,
    SupportDepth.DEEP: 5,
}

class PolicyEngine:
    def resolve_phase(
        self,
        current_phase,
        predicted_phase,
        confidence,
        previous_frustration: str | None = None,
    ):
        if predicted_phase == current_phase:
            return current_phase

        if predicted_phase not in TRANSITIONS[current_phase]:
            return current_phase

        # If the student was highly frustrated last turn, require a stronger
        # confidence signal before switching phase. A frustrated student saying
        # "I give up" reads like re-orientation but is usually still PERFORMANCE.
        threshold = SWITCH_THRESHOLD
        if previous_frustration == "HIGH":
            threshold = 0.85

        if confidence < threshold:
            return current_phase

        return predicted_phase

    def allowed_support_levels(self, checkpoint: Checkpoint) -> Set[SupportLevel]:
        if checkpoint.context_gap == ContextGap.CRITICAL:
            return {SupportLevel.CLARIFY}

        if checkpoint.srl_focus == SRLFocus.REFLECT:
            return {SupportLevel.REFLECT, SupportLevel.EVALUATION}

        if checkpoint.frustration_level == FrustrationLevel.HIGH:
            return {
                SupportLevel.EXPLAIN,
                SupportLevel.PARTIAL,
            }

        if checkpoint.frustration_level == FrustrationLevel.MEDIUM:
            return {
                SupportLevel.QUESTION,
                SupportLevel.HINT,
                SupportLevel.STRUCTURE,
                SupportLevel.EXPLAIN,
            }

        if checkpoint.expertise_level == ExpertiseLevel.NOVICE:
            return {
                SupportLevel.CLARIFY,
                SupportLevel.QUESTION,
                SupportLevel.HINT,
                SupportLevel.STRUCTURE,
                SupportLevel.EXPLAIN,
                SupportLevel.PARTIAL,
            }

        if checkpoint.srl_focus == SRLFocus.GOAL:
            return {
                SupportLevel.QUESTION,
                SupportLevel.STRUCTURE,
                SupportLevel.CLARIFY,
            }

        if checkpoint.srl_focus == SRLFocus.STRATEGY:
            return {
                SupportLevel.QUESTION,
                SupportLevel.HINT,
                SupportLevel.STRUCTURE,
                SupportLevel.PARTIAL,
            }

        if checkpoint.srl_focus == SRLFocus.MONITOR:
            return {
                SupportLevel.QUESTION,
                SupportLevel.HINT,
                SupportLevel.EXPLAIN,
                SupportLevel.STRUCTURE,
            }

        return set(SupportLevel)

    def fallback_support_level(self, checkpoint: Checkpoint) -> SupportLevel:
        if checkpoint.context_gap == ContextGap.CRITICAL:
            return SupportLevel.CLARIFY

        if checkpoint.srl_focus == SRLFocus.REFLECT:
            if checkpoint.progress_state == ProgressState.DONEISH:
                return SupportLevel.EVALUATION
            return SupportLevel.REFLECT

        if checkpoint.frustration_level == FrustrationLevel.HIGH:
            if checkpoint.progress_state == ProgressState.STALLED:
                return SupportLevel.PARTIAL
            return SupportLevel.STRUCTURE

        if checkpoint.srl_focus == SRLFocus.GOAL:
            return SupportLevel.QUESTION

        if checkpoint.srl_focus == SRLFocus.STRATEGY:
            return SupportLevel.HINT

        if checkpoint.srl_focus == SRLFocus.MONITOR:
            return SupportLevel.HINT

        return SupportLevel.QUESTION

    def _next_support_level(
        self,
        current: SupportLevel,
        allowed: Set[SupportLevel],
        checkpoint: Checkpoint,
    ) -> SupportLevel:
        progression_order = [
            SupportLevel.CLARIFY,
            SupportLevel.QUESTION,
            SupportLevel.HINT,
            SupportLevel.STRUCTURE,
            SupportLevel.EXPLAIN,
            SupportLevel.PARTIAL,
        ]

        if checkpoint.srl_focus == SRLFocus.REFLECT:
            if current == SupportLevel.REFLECT and SupportLevel.EVALUATION in allowed:
                return SupportLevel.EVALUATION
            return current

        if current in progression_order:
            idx = progression_order.index(current)
            for next_idx in range(idx + 1, len(progression_order)):
                candidate = progression_order[next_idx]
                if candidate in allowed:
                    return candidate

        return current

    def _enforce_depth_floor(
        self,
        depth: SupportDepth,
        expertise_level: ExpertiseLevel,
    ) -> SupportDepth:
        floor = _DEPTH_FLOOR.get(expertise_level, SupportDepth.SUBSTANTIVE)
        return depth if _DEPTH_ORDER[depth] >= _DEPTH_ORDER[floor] else floor

    def enforce_decision(
        self,
        checkpoint: Checkpoint,
        decision: Decision,
        recent_support_levels: list[SupportLevel],
    ) -> Decision:
        allowed = self.allowed_support_levels(checkpoint)

        support_level = (
            decision.support_level
            if decision.support_level in allowed
            else self.fallback_support_level(checkpoint)
        )

        repeated = (
            len(recent_support_levels) >= 2
            and recent_support_levels[-1] == support_level
            and recent_support_levels[-2] == support_level
        )

        if repeated:
            support_level = self._next_support_level(
                current=support_level,
                allowed=allowed,
                checkpoint=checkpoint,
            )

        if (
            support_level == SupportLevel.QUESTION
            and not checkpoint.has_attempt
            and checkpoint.progress_state in {ProgressState.MOVING, ProgressState.STALLED}
            and recent_support_levels[-1:] == [SupportLevel.QUESTION]
            and SupportLevel.HINT in allowed
        ):
            support_level = SupportLevel.HINT

        if (
            support_level == SupportLevel.HINT
            and checkpoint.progress_state == ProgressState.STALLED
            and recent_support_levels[-1:] == [SupportLevel.HINT]
            and SupportLevel.STRUCTURE in allowed
        ):
            support_level = SupportLevel.STRUCTURE

        if support_level not in CODE_ALLOWED_LEVELS:
            can_show_code = False
        else:
            can_show_code = (
                checkpoint.frustration_level in {FrustrationLevel.MEDIUM, FrustrationLevel.HIGH}
                or checkpoint.progress_state == ProgressState.STALLED
                or checkpoint.expertise_level == ExpertiseLevel.NOVICE
            )

        must_end_with_question = support_level in QUESTION_REQUIRED_LEVELS

        should_request_attempt = decision.should_request_attempt
        if not checkpoint.has_attempt and support_level in {
            SupportLevel.CLARIFY,
            SupportLevel.QUESTION,
            SupportLevel.HINT,
            SupportLevel.STRUCTURE,
        }:
            should_request_attempt = True

        # --- Depth floor enforcement ---
        # The LLM sets support_depth, but we hard-override it here if it falls
        # below the minimum for the student's expertise level.  This is the
        # single place where depth calibration is guaranteed — downstream
        # generation reads this enforced value from the decision payload.
        enforced_depth = self._enforce_depth_floor(
            depth=decision.support_depth,
            expertise_level=checkpoint.expertise_level,
        )

        return replace(
            decision,
            support_level=support_level,
            support_depth=enforced_depth,
            response_prompt_file=response_prompt_file_for(support_level),
            can_show_code=can_show_code,
            must_end_with_question=must_end_with_question,
            should_request_attempt=should_request_attempt,
        )