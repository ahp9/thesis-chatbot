import asyncio
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict

from lib.enums import Phase, TutorMode
from services.llm_client import get_client
from services.orchestrator import Orchestrator
from services.tutor import run_basic_tutor
from utils.logger import save_conversation
from utils.message import (
    build_combined_user_content,
    merge_db_steps_with_transcript,
    persist_uploaded_elements_for_message,
)
from utils.session import coerce_phase, load_conversation, save_transcript_if_possible
from utils.storage import LocalStorageClient

sqlite3.register_adapter(list, lambda lst: json.dumps(lst))
sqlite3.register_adapter(dict, lambda dct: json.dumps(dct))

client = get_client()
orchestrator = Orchestrator(client)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

MOCK_USERS = {
    "student1@research.local": "password123",
    "student2@research.local": "study2026",
    "student3@research.local": "research_mode",
    "working_mode@admin.local": "working_mode",
    "user_1@usability_test_1.local": "usability1",
    "user_2@usability_test_2.local": "usability2",
    "user_3@usability_test_3.local": "usability3",
    "user_3@usability_test_4.local": "usability4",
    "user_5@usability_test_5.local": "usability5",
    "user_1_2@usability_test_1_2.local": "usability1_2",
    "user_2_2@usability_test_2_2.local": "usability2_2",
    "user_3_2@usability_test_3_2.local": "usability3_2",
    "user_4_2@usability_test_4.local": "usability4_2",
    "user_5_2@usability_test_5_2.local": "usability5_2",

    "pilot_1@experiment.local": "pilot1",
    "test_1@experiment.local": "test1",

    "participant_1@experiment.local": "experiment1",
    "participant_2@experiment.local": "experiment2",
    "participant_3@experiment.local": "experiment3",
    "participant_4@experiment.local": "experiment4",
    "participant_5@experiment.local": "experiment5",
    "participant_6@experiment.local": "experiment6",
    "participant_7@experiment.local": "experiment7",
    "participant_8@experiment.local": "experiment8",
    "participant_9@experiment.local": "experiment9",
    "participant_10@experiment.local": "experiment10",
    "participant_11@experiment.local": "experiment11",
    "participant_12@experiment.local": "experiment12",
}

UPLOAD_ROOT = Path("./uploaded_files")


@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """Authenticate a user against the static MOCK_USERS credential map."""
    if username in MOCK_USERS and MOCK_USERS[username] == password:
        return cl.User(
            identifier=username,
            metadata={
                "display_name": username.split("@")[0],
                "provider": "credentials",
            },
        )
    return None


@cl.data_layer
def get_data_layer():
    """Configure SQLite persistence with a local file storage provider."""
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    return SQLAlchemyDataLayer(
        conninfo="sqlite+aiosqlite:///./chainlit.db",
        storage_provider=LocalStorageClient(UPLOAD_ROOT),
    )


@cl.set_chat_profiles
async def chat_profile(_current_user: cl.User | None) -> list[cl.ChatProfile]:
    """Return the two available chat profiles: SRL (Group A) and Basic (Group B)."""
    return [
        cl.ChatProfile(
            name=TutorMode.SRL.value,
            markdown_description="Group A",
        ),
        cl.ChatProfile(
            name=TutorMode.BASIC.value,
            markdown_description="Group B",
        ),
    ]


@cl.on_chat_start
async def start():
    """Initialise session state and restore any existing transcript on chat start."""
    user = cl.user_session.get("user")
    student_id = user.identifier.split("@")[0]
    tutor_type = cl.user_session.get("chat_profile") or TutorMode.SRL.value
    thread_id = cl.context.session.thread_id

    cl.user_session.set("user_id", student_id)
    cl.user_session.set("tutor_type", tutor_type)
    cl.user_session.set("session_id", thread_id)
    cl.user_session.set("current_phase", Phase.FORETHOUGHT.value)

    # Restore history if session exists
    saved = load_conversation(student_id, thread_id)
    if saved and isinstance(saved.get("history"), list):
        llm_history = saved["history"]
        cl.user_session.set("llm_history", llm_history)

        for item in reversed(llm_history):
            if item.get("role") == "assistant" and isinstance(item.get("route"), dict):
                cl.user_session.set("current_phase", item["route"].get("phase", Phase.FORETHOUGHT.value))
                break

        logger.info(
            "Restored existing transcript on chat start | user=%s | thread=%s | turns=%d",
            student_id, thread_id, len(llm_history),
        )
    else:
        cl.user_session.set("llm_history", [])
        logger.info("Chat started | user=%s | thread=%s | mode=%s", student_id, thread_id, tutor_type)


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Restore session state from the DB thread and merge it with the saved transcript."""
    raw_metadata = thread.get("metadata", {})
    metadata: dict[str, Any]

    if isinstance(raw_metadata, str):
        try:
            metadata = json.loads(raw_metadata)
        except json.JSONDecodeError:
            metadata = {}
    elif isinstance(raw_metadata, dict):
        metadata = raw_metadata
    else:
        metadata = {}

    student_id = metadata.get("user_id", "Unknown")
    tutor_type = metadata.get("tutor_type", TutorMode.SRL.value)
    session_id = thread.get("id")

    cl.user_session.set("user_id", student_id)
    cl.user_session.set("tutor_type", tutor_type)
    cl.user_session.set("session_id", session_id)

    # Merge DB turns with saved SRL metadata
    steps = thread.get("steps", [])
    saved = load_conversation(student_id, session_id)
    saved_history = saved.get("history", []) if saved else []

    llm_history = merge_db_steps_with_transcript(steps, saved_history)

    # Persist merged history so on_chat_start sees it on next load
    if student_id != "working_mode" and llm_history:
        save_conversation(
            session_id=session_id,
            user_id=student_id,
            tutor_type=tutor_type,
            history=llm_history,
        )

    cl.user_session.set("llm_history", llm_history)

    # Recover last known phase
    current_phase = metadata.get("current_phase") or Phase.FORETHOUGHT.value
    for item in reversed(llm_history):
        if item.get("role") == "assistant" and isinstance(item.get("route"), dict):
            current_phase = item["route"].get("phase", Phase.FORETHOUGHT.value)
            break

    cl.user_session.set("current_phase", current_phase)
    logger.info(
        "Resumed thread | user=%s | session=%s | db_steps=%d | transcript_turns=%d | merged=%d",
        student_id, session_id, len(steps), len(saved_history), len(llm_history),
    )


@cl.on_message
async def main(message: cl.Message):
    """Handle an incoming student message, route it through the active tutor pipeline, and stream the reply."""
    tutor_type = str(cl.user_session.get("tutor_type") or TutorMode.SRL.value)
    session_id = cl.user_session.get("session_id")
    student_id = cl.user_session.get("user_id")
    llm_history: list[dict[str, Any]] = cl.user_session.get("llm_history") or []
    current_phase = coerce_phase(cl.user_session.get("current_phase"))

    combined_user_content, uploaded_files_meta = build_combined_user_content(message)

    ai_text = ""
    prefix = ""

    # SRL-mode metadata — only populated in SRL mode.
    route_for_history: dict[str, Any] = {
        "phase": current_phase.value,
        "strategy": "NONE",
        "confidence": 0.0,
        "signals": [],
    }
    checkpoint_for_history = None
    decision_for_history = None
    safety_for_history = None
    draft_reply_for_history = ""

    is_srl = tutor_type == TutorMode.SRL.value

    async with cl.Step(name="Thinking...") as step:

        if is_srl:
            # SRL: full pipeline — route → classify → generate → safety
            result = await orchestrator.handle_turn(
                user_message=combined_user_content,
                llm_history=llm_history,
                current_phase=current_phase,
            )

            ai_text = result.reply
            prefix = result.prefix
            draft_reply_for_history = result.draft_reply
            route_for_history = result.route.to_dict()
            checkpoint_for_history = result.control.checkpoint.to_dict()
            decision_for_history = result.control.decision.to_dict()
            safety_for_history = result.safety.to_dict()

            cl.user_session.set("current_phase", result.route.phase.value)

            logger.info("=" * 40)
            logger.info("SRL SESSION: %s | USER: %s", session_id, student_id)
            logger.info(
                "PHASE: %s (Confidence: %.2f) | SRL Signals: %s",
                result.route.phase.value,
                result.route.confidence,
                result.route.srl_signal,
            )
            logger.info(
                "CHECKPOINT: Kind=%s | Stage=%s | Progress=%s | Attempt=%s | Gap=%s | SRL=%s | SubtaskScope=%s",
                result.control.checkpoint.request_kind.value,
                result.control.checkpoint.task_stage.value,
                result.control.checkpoint.progress_state.value,
                result.control.checkpoint.has_attempt,
                result.control.checkpoint.context_gap.value,
                result.control.checkpoint.srl_focus.value,
                result.control.checkpoint.subtask_scope,
            )
            logger.info(
                "LEARNER: Expertise=%s | Frustration=%s | SRL=%s",
                result.control.checkpoint.expertise_level.value,
                result.control.checkpoint.frustration_level.value,
                result.control.checkpoint.srl_focus.value,
            )
            logger.info(
                "DECISION: Level=%s | Depth=%s | CanShowCode=%s",
                result.control.decision.support_level.value,
                result.control.decision.support_depth.value,
                result.control.decision.can_show_code,
            )
            logger.info(
                "SAFETY CHECK: skipped=%s | is_safe=%s | leaks_solution=%s | reason=%s",
                result.safety.was_skipped,
                result.safety.is_safe,
                result.safety.leaks_solution,
                result.safety.reason,
            )
            logger.info("=" * 40)

        else:
            # Basic: single LLM call, no classification or safety chain
            logger.info("BASIC SESSION: %s | USER: %s", session_id, student_id)
            ai_text = await run_basic_tutor(
                llm_history,
                combined_user_content,
            )

    # Stream reply and update history
    msg = cl.Message(content="")
    await msg.send()

    if prefix:
        await msg.stream_token(prefix)

    for chunk in ai_text.split(" "):
        await msg.stream_token(chunk + " ")
        await asyncio.sleep(0.01)

    await msg.update()
    await step.remove()

    # Persist uploaded files as message elements in Chainlit storage.
    persisted_files = await persist_uploaded_elements_for_message(message, msg.id)

    # Update history
    llm_history.append(
        {
            "role": "user",
            "content": combined_user_content,
            "timestamp": datetime.now().isoformat(),
            "uploaded_files": uploaded_files_meta,
        }
    )

    history_entry: dict[str, Any] = {
        "role": "assistant",
        "content": ai_text,
        "timestamp": datetime.now().isoformat(),
        "draft_reply": draft_reply_for_history or ai_text,
        "persisted_files": persisted_files,
    }

    # Only attach SRL metadata when it was actually computed.
    if is_srl:
        history_entry["route"] = route_for_history
        history_entry["diagnosis"] = checkpoint_for_history
        history_entry["decision"] = decision_for_history
        history_entry["check"] = safety_for_history

    llm_history.append(history_entry)
    cl.user_session.set("llm_history", llm_history)
    save_transcript_if_possible(session_id, student_id, tutor_type, llm_history)


@cl.on_chat_end
async def end():
    """Save the final transcript when the chat session ends."""
    session_id = cl.user_session.get("session_id")
    student_id = cl.user_session.get("user_id")
    tutor_type = cl.user_session.get("tutor_type") or TutorMode.SRL.value
    llm_history = cl.user_session.get("llm_history")

    save_transcript_if_possible(
        session_id=session_id,
        student_id=student_id,
        tutor_type=tutor_type,
        llm_history=llm_history,
    )
