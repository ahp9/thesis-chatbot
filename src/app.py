import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, Optional

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.element import File
from chainlit.types import ThreadDict

from services.llm_client import get_client
from services.router import route_message, update_phase
from services.srl_chain import run_srl_chain
from services.tutor import build_system_prompt, run_tutor
from utils.file import read_uploaded_file
from utils.logger import save_conversation

# SQLite adapters are unchanged from your current app file.
sqlite3.register_adapter(list, lambda lst: json.dumps(lst))
sqlite3.register_adapter(dict, lambda dct: json.dumps(dct))

client = get_client()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MOCK_USERS = {
    "student1@research.local": "password123",
    "student2@research.local": "study2026",
    "student3@research.local": "research_mode",
    "working_mode@admin.local": "working_mode",
    "user_1@usability_test_1.local": "usability1",
    "user_2@usability_test_2.local": "usability2",
    "user_3@usability_test_3.local": "usability3",
}

MAX_CHARS = 80_000


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
            name="SRL Tutor",
            markdown_description="Phase-aware chained tutoring with pushback.",
        ),
        cl.ChatProfile(
            name="Basic Tutor",
            markdown_description="Direct answers and code support.",
        ),
    ]


@cl.on_chat_start
async def start():
    user = cl.user_session.get("user")
    student_id = user.identifier.split("@")[0]
    tutor_type = cl.user_session.get("chat_profile")
    thread_id = cl.context.session.thread_id

    cl.user_session.set("user_id", student_id)
    cl.user_session.set("tutor_type", tutor_type)
    cl.user_session.set("session_id", thread_id)
    cl.user_session.set("llm_history", [])
    cl.user_session.set("current_phase", "FORETHOUGHT")


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
    tutor_type = metadata.get("tutor_type", "SRL Tutor")

    cl.user_session.set("user_id", student_id)
    cl.user_session.set("tutor_type", tutor_type)
    cl.user_session.set("session_id", thread.get("id"))

    steps = thread.get("steps", [])
    llm_history = []
    for step in steps:
        role = (
            "assistant" if step.get("type") == 
            "assistant_message" else "user"
        )
        content = step.get("output") or step.get("input") or ""
        llm_history.append({"role": role, "content": content})

    cl.user_session.set("llm_history", llm_history)
    cl.user_session.set(
        "current_phase", metadata.get("current_phase", "FORETHOUGHT")
    )


def maybe_save(session_id: str, student_id: str, tutor_type: str, llm_history):
    if student_id != "working_mode" and llm_history:
        save_conversation(
            session_id=session_id,
            user_id=student_id,
            tutor_type=tutor_type,
            history=llm_history,
        )


@cl.on_message
async def main(message: cl.Message):
    tutor_type = cl.user_session.get("tutor_type")
    session_id = cl.user_session.get("session_id")
    student_id = cl.user_session.get("user_id")
    llm_history: list[dict[str, Any]] = (
        cl.user_session.get("llm_history") or []
    )
    current_phase = cl.user_session.get("current_phase")

    file_text_blocks = []
    if message.elements:
        for el in message.elements:
            if isinstance(el, File) and getattr(el, "path", None):
                try:
                    content = read_uploaded_file(el)
                    content = content[:MAX_CHARS]
                    file_text_blocks.append(
                        f"--- FILE: {el.name} ({el.mime}) ---\n{content}\n"
                        f"--- END FILE ---"
                    )
                except Exception as exc:
                    file_text_blocks.append(
                        f"--- FILE: {el.name} ---\n"
                        f"[Error reading file: {exc}]\n"
                        f"--- END FILE ---"
                    )

    combined_user_content = message.content or ""
    if file_text_blocks:
        combined_user_content += "\n\n" + "\n\n".join(file_text_blocks)

    llm_history.append(
        {
            "role": "user",
            "content": combined_user_content,
            "timestamp": datetime.now().isoformat(),
        }
    )

    route: Dict[str, Any] = {
        "phase": current_phase,
        "strategy": "NONE",
        "confidence": 0.0,
    }
    if tutor_type == "SRL Tutor":
        route = await route_message(
            client, message.content, llm_history, current_phase
        )
        predicted_phase = route.get("phase", current_phase)
        confidence = route.get("confidence", 0.0)
        new_phase = update_phase(current_phase, 
                                    predicted_phase, confidence)
        cl.user_session.set("current_phase", new_phase)
        route["phase"] = new_phase

    cl.user_session.set("last_route", route)

    if tutor_type == "SRL Tutor":
        chain_result = await run_srl_chain(
            client, route, llm_history, message.content or ""
        )
        ai_text = chain_result["reply"]
        llm_history.append(
            {
                "role": "assistant",
                "content": ai_text,
                "timestamp": datetime.now().isoformat(),
                "route": route,
                "diagnosis": chain_result["diagnosis"],
                "decision": chain_result["decision"],
                "check": chain_result["check"],
                "draft_reply": chain_result["reply"],
            }
        )
    else:
        system_prompt = build_system_prompt(tutor_type, route)
        ai_text = await run_tutor(client, system_prompt, llm_history)
        llm_history.append(
            {
                "role": "assistant",
                "content": ai_text,
                "timestamp": datetime.now().isoformat(),
                "system_prompt": system_prompt,
            }
        )

    cl.user_session.set("llm_history", llm_history)
    maybe_save(
        session_id=session_id,
        student_id=student_id,
        tutor_type=tutor_type,
        llm_history=llm_history,
    )
    await cl.Message(content=ai_text).send()


@cl.on_chat_end
async def end():
    session_id = cl.user_session.get("session_id")
    student_id = cl.user_session.get("user_id")
    tutor_type = cl.user_session.get("tutor_type") or "SRL Tutor"
    llm_history = cl.user_session.get("llm_history")
    maybe_save(
        session_id=session_id,
        student_id=student_id,
        tutor_type=tutor_type,
        llm_history=llm_history,
    )
