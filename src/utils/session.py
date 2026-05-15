import json
import logging
import os
from typing import Any

from lib.enums import Phase
from utils.logger import save_conversation

LOG_FILE = "transcripts"

logger = logging.getLogger(__name__)


def get_log_filename(user_id, session_id):
    """Return the transcript file path for a given user and session."""
    os.makedirs(LOG_FILE, exist_ok=True)
    return os.path.join(LOG_FILE, f"user_{user_id}_session_{session_id}.json")


def load_conversation(user_id, session_id):
    """Load and return the saved transcript JSON, or None if missing or unreadable."""
    filename = get_log_filename(user_id, session_id)
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Could not load transcript %s: %s", filename, exc)
        return None


def save_transcript_if_possible(session_id: str, student_id: str, tutor_type: str, llm_history):
    """Save the conversation transcript, skipping working-mode sessions and empty histories."""
    if student_id != "working_mode" and llm_history:
        logger.info("Saving transcript...")
        save_conversation(
            session_id=session_id,
            user_id=student_id,
            tutor_type=tutor_type,
            history=llm_history,
        )


def coerce_phase(value: Any) -> Phase:
    """Parse a phase string into a Phase enum, defaulting to FORETHOUGHT on failure."""
    try:
        return Phase(str(value or "FORETHOUGHT").upper())
    except ValueError:
        return Phase.FORETHOUGHT
