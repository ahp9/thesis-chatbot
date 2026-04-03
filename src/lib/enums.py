from enum import Enum, IntEnum


class TutorMode(str, Enum):
    SRL = "SRL Tutor"
    BASIC = "Basic Tutor"


class Phase(str, Enum):
    FORETHOUGHT = "FORETHOUGHT"
    PERFORMANCE = "PERFORMANCE"
    REFLECTION = "REFLECTION"


class SupportLevel(str, Enum):
    CLARIFY = "CLARIFY"
    QUESTION = "QUESTION"
    HINT = "HINT"
    STRUCTURE = "STRUCTURE"
    EXPLAIN = "EXPLAIN"
    PARTIAL = "PARTIAL"
    REFLECT = "REFLECT"
    EVALUATION = "EVALUATION"


class RequestKind(str, Enum):
    RESOURCE = "RESOURCE"
    PRODUCT = "PRODUCT"


class ContextGap(str, Enum):
    NONE = "NONE"
    SMALL = "SMALL"
    CRITICAL = "CRITICAL"


class ExpertiseLevel(str, Enum):
    NOVICE = "NOVICE"
    INTERMEDIATE = "INTERMEDIATE"
    ADVANCED = "ADVANCED"


class FrustrationLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TaskStage(str, Enum):
    STARTING = "STARTING"
    WORKING = "WORKING"
    REVIEWING = "REVIEWING"


class ProgressState(str, Enum):
    MOVING = "MOVING"
    STALLED = "STALLED"
    DONEISH = "DONEISH"


class SRLFocus(str, Enum):
    GOAL = "GOAL"
    STRATEGY = "STRATEGY"
    MONITOR = "MONITOR"
    REFLECT = "REFLECT"
    NONE = "NONE"


class SupportDepth(str, Enum):
    SURFACE = "SURFACE"
    SURFACE_PLUS = "SURFACE_PLUS"
    SUBSTANTIVE = "SUBSTANTIVE"
    SUBSTANTIVE_PLUS = "SUBSTANTIVE_PLUS"
    DEEP = "DEEP"
