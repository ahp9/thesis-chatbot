import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Optional

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.element import File
from chainlit.types import ThreadDict

from lib.enums import Phase, TutorMode
from services.llm_client import get_client
from services.orchestrator import Orchestrator
from services.tutor import build_system_prompt, run_tutor
from utils.file import read_uploaded_file
from utils.logger import save_conversation

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
}

MAX_CHARS = 80_000
LOG_FILE = "transcripts"


def get_log_filename(user_id, session_id):
    return os.path.join(LOG_FILE, f"user_{user_id}_session_{session_id}.jsonl")


def load_conversation(user_id, session_id):
    filename = get_log_filename(user_id, session_id)
    if not os.path.exists(filename):
        return None

    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Could not load transcript %s: %s", filename, exc)
        return None


def maybe_save(session_id: str, student_id: str, tutor_type: str, llm_history):
    if student_id != "working_mode" and llm_history:
        save_conversation(
            session_id=session_id,
            user_id=student_id,
            tutor_type=tutor_type,
            history=llm_history,
        )


def _coerce_phase(value: Any) -> Phase:
    try:
        return Phase(str(value or "FORETHOUGHT").upper())
    except ValueError:
        return Phase.FORETHOUGHT


def _build_combined_user_content(message: cl.Message) -> str:
    file_text_blocks: list[str] = []

    if message.elements:
        for el in message.elements:
            if isinstance(el, File) and getattr(el, "path", None):
                try:
                    content = read_uploaded_file(el)
                    content = content[:MAX_CHARS]
                    file_text_blocks.append(
                        f"--- FILE: {el.name} ({el.mime}) --- CONTENT START ---\n"
                        f"{content}\n--- END FILE ---"
                    )
                except Exception as exc:
                    file_text_blocks.append(f"[Error reading file {el.name}: {exc}]")

    combined_user_content = message.content or ""
    if file_text_blocks:
        combined_user_content += "\n\n" + "\n\n".join(file_text_blocks)

    return combined_user_content


@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> Optional[cl.User]:
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
    return SQLAlchemyDataLayer(conninfo="sqlite+aiosqlite:///./chainlit.db")


@cl.set_chat_profiles
async def chat_profile(current_user: cl.User | None) -> list[cl.ChatProfile]:
    return [
        cl.ChatProfile(
            name=TutorMode.SRL.value,
            markdown_description="Phase-aware chained tutoring with pushback.",
        ),
        cl.ChatProfile(
            name=TutorMode.BASIC.value,
            markdown_description="Direct answers and code support.",
        ),
    ]


@cl.on_chat_start
async def start():
    user = cl.user_session.get("user")
    student_id = user.identifier.split("@")[0]
    tutor_type = cl.user_session.get("chat_profile") or TutorMode.SRL.value
    thread_id = cl.context.session.thread_id

    cl.user_session.set("user_id", student_id)
    cl.user_session.set("tutor_type", tutor_type)
    cl.user_session.set("session_id", thread_id)
    cl.user_session.set("llm_history", [])
    cl.user_session.set("current_phase", Phase.FORETHOUGHT.value)


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
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

    saved = load_conversation(student_id, session_id)

    if saved and isinstance(saved.get("history"), list):
        llm_history = saved["history"]
        cl.user_session.set("llm_history", llm_history)

        current_phase = metadata.get("current_phase")
        if not current_phase:
            current_phase = Phase.FORETHOUGHT.value
            for item in reversed(llm_history):
                if item.get("role") == "assistant" and isinstance(item.get("route"), dict):
                    current_phase = item["route"].get("phase", Phase.FORETHOUGHT.value)
                    break

        cl.user_session.set("current_phase", current_phase)
        logger.info("Resumed saved transcript for %s, session %s", student_id, session_id)
        return

    steps = thread.get("steps", [])
    llm_history = []

    for step in steps:
        role = "assistant" if step.get("type") == "assistant_message" else "user"
        content = step.get("output") or step.get("input") or ""

        if content and content != "{}":
            llm_history.append({"role": role, "content": content})

    cl.user_session.set("llm_history", llm_history)
    cl.user_session.set(
        "current_phase",
        metadata.get("current_phase", Phase.FORETHOUGHT.value),
    )


@cl.on_message
async def main(message: cl.Message):
    tutor_type = str(cl.user_session.get("tutor_type") or TutorMode.SRL.value)
    session_id = cl.user_session.get("session_id")
    student_id = cl.user_session.get("user_id")
    llm_history: list[dict[str, Any]] = cl.user_session.get("llm_history") or []
    current_phase = _coerce_phase(cl.user_session.get("current_phase"))

    combined_user_content = _build_combined_user_content(message)

    ai_text = ""
    prefix = ""
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

    async with cl.Step(name="Tutor is thinking...") as step:
        if tutor_type == TutorMode.SRL.value:
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
                "PHASE: %s (Confidence: %.2f)",
                result.route.phase.value,
                result.route.confidence,
            )
            logger.info(
                "CHECKPOINT: Kind=%s | Stage=%s | Progress=%s | Attempt=%s | Gap=%s",
                result.control.checkpoint.request_kind.value,
                result.control.checkpoint.task_stage.value,
                result.control.checkpoint.progress_state.value,
                result.control.checkpoint.has_attempt,
                result.control.checkpoint.context_gap.value,
            )
            logger.info(
                "LEARNER: Expertise=%s | Frustration=%s | SRL=%s | ImplAllowed=%s",
                result.control.checkpoint.expertise_level.value,
                result.control.checkpoint.frustration_level.value,
                result.control.checkpoint.srl_focus.value,
                result.control.checkpoint.implementation_allowed,
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
            system_prompt = build_system_prompt(tutor_type, route_for_history)
            ai_text = await run_tutor(client, system_prompt, llm_history)

    msg = cl.Message(content="")

    if prefix:
        await msg.stream_token(prefix)

    for chunk in ai_text.split(" "):
        await msg.stream_token(chunk + " ")
        await asyncio.sleep(0.01)

    await msg.send()
    await step.remove()

    llm_history.append(
        {
            "role": "user",
            "content": combined_user_content,
            "timestamp": datetime.now().isoformat(),
        }
    )

    history_entry: dict[str, Any] = {
        "role": "assistant",
        "content": ai_text,
        "timestamp": datetime.now().isoformat(),
        "route": route_for_history,
        "diagnosis": checkpoint_for_history,
        "decision": decision_for_history,
        "check": safety_for_history,
        "draft_reply": draft_reply_for_history or ai_text,
    }

    llm_history.append(history_entry)
    cl.user_session.set("llm_history", llm_history)
    maybe_save(session_id, student_id, tutor_type, llm_history)


@cl.on_chat_end
async def end():
    session_id = cl.user_session.get("session_id")
    student_id = cl.user_session.get("user_id")
    tutor_type = cl.user_session.get("tutor_type") or TutorMode.SRL.value
    llm_history = cl.user_session.get("llm_history")

    maybe_save(
        session_id=session_id,
        student_id=student_id,
        tutor_type=tutor_type,
        llm_history=llm_history,
    )