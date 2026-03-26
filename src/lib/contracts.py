from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, Dict, List

from lib.enums import (
    ContextGap,
    ExpertiseLevel,
    FrustrationLevel,
    Phase,
    ProgressState,
    RequestKind,
    SRLFocus,
    SupportDepth,
    SupportLevel,
    TaskStage,
)


def _normalize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {k: _normalize(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    return value


class SerializableContract:
    def to_dict(self) -> Dict[str, Any]:
        return _normalize(self)


@dataclass
class RouteResult(SerializableContract):
    phase: Phase
    strategy: str
    confidence: float
    signals: List[str]


@dataclass
class Checkpoint(SerializableContract):
    request_kind: RequestKind
    task_stage: TaskStage
    progress_state: ProgressState
    has_attempt: bool
    context_gap: ContextGap
    expertise_level: ExpertiseLevel
    frustration_level: FrustrationLevel
    srl_focus: SRLFocus
    implementation_allowed: bool
    confidence: float
    rationale: List[str]
    parse_ok: bool = True


@dataclass
class Decision(SerializableContract):
    support_level: SupportLevel
    response_prompt_file: str
    support_depth: SupportDepth
    can_show_code: bool
    must_end_with_question: bool
    should_request_attempt: bool
    confidence: float
    rationale: List[str]
    parse_ok: bool = True


@dataclass
class CombinedControlResult(SerializableContract):
    checkpoint: Checkpoint
    decision: Decision


@dataclass
class SafetyResult(SerializableContract):
    is_safe: bool
    leaks_solution: bool
    skipped_diagnosis: bool
    reason: str
    was_skipped: bool = False


@dataclass
class TurnResult(SerializableContract):
    reply: str
    draft_reply: str
    route: RouteResult
    control: CombinedControlResult
    safety: SafetyResult
    was_rewritten: bool
    prefix: str = ""