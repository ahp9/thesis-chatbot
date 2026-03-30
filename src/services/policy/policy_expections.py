class PolicyError(Exception):
    """Base exception for pedagogy policy errors."""


class InvalidPhaseTransitionError(PolicyError):
    """Raised when a phase transition is blocked by the controller."""


class InvalidSupportDecisionError(PolicyError):
    """Raised when a support decision violates deterministic policy."""