from __future__ import annotations

from lib.contracts import RouteResult
from lib.enums import Phase
from services.router import route_message


class RouterService:
    def __init__(self, client):
        self.client = client

    async def route(
        self,
        user_message: str,
        llm_history: list[dict],
        current_phase: Phase,
    ) -> RouteResult:

        data = await route_message(
            self.client,
            user_message,
            llm_history,
            current_phase.value,
        )

        raw_phase = (data.get("phase") or current_phase.value).upper()
        try:
            phase = Phase(raw_phase)
        except ValueError:
            phase = current_phase

        return RouteResult(
            phase=phase,
            confidence=float(data.get("confidence", 0.0) or 0.0),
            srl_signal=str(data.get("srl_signal") or "NONE").upper(),
            signals=list(data.get("signals", []) or []),
            trajectory_note=str(data.get("trajectory_note") or ""),
        )