from __future__ import annotations

import logging
from typing import Any, Dict

from lib.contracts import _normalize

logger = logging.getLogger(__name__)


class Telemetry:
    def event(self, name: str, payload: Dict[str, Any]) -> None:
        logger.info("[EVENT] %s | %s", name, _normalize(payload))