import asyncio
import json
import os
import logging
import sqlite3
from datetime import datetime
from typing import Any, Optional, cast

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.element import File
from chainlit.types import ThreadDict

from services.llm_client import get_client
from services.router import route_message, update_phase
from services.srl_chain import (
    check_reply,
    checkpoint_and_decide, 
    generate_full_reply, 
    rewrite_reply
)
from services.tutor import build_system_prompt, run_tutor
from utils.file import read_uploaded_file
from utils.logger import save_conversation

sqlite3.register_adapter(list, lambda lst: json.dumps(lst))
sqlite3.register_adapter(dict, lambda dct: json.dumps(dct))

client = get_client()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True # This 'force' is critical to override Chainlit's default logger
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
        logger.error(f"Could not load transcript {filename}: {exc}")
        return None


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
            current_phase = "FORETHOUGHT"
            for item in reversed(llm_history):
                if item.get("role") == "assistant" and isinstance(item.get("route"), dict):
                    current_phase = item["route"].get("phase", "FORETHOUGHT")
                    break

        cl.user_session.set("current_phase", current_phase)
        logger.info(f"Resumed saved transcript for {student_id}, session {session_id}")
        return

    # fallback only if no transcript file exists
    steps = thread.get("steps", [])
    llm_history = []

    for step in steps:
        role = "assistant" if step.get("type") == "assistant_message" else "user"
        content = step.get("output") or step.get("input") or ""

        if content and content != "{}":
            llm_history.append({"role": role, "content": content})

    cl.user_session.set("llm_history", llm_history)
    cl.user_session.set("current_phase", metadata.get("current_phase", "FORETHOUGHT"))

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
    llm_history: list[dict[str, Any]] = cl.user_session.get("llm_history") or []
    current_phase = cast(str, cl.user_session.get("current_phase") or "FORETHOUGHT")

    # 1. Process files
    file_text_blocks = []
    if message.elements:
        for el in message.elements:
            if isinstance(el, File) and getattr(el, "path", None):
                try:
                    content = read_uploaded_file(el)
                    content = content[:MAX_CHARS] 
                    file_text_blocks.append(
                        f"--- FILE: {el.name} ({el.mime}) --- CONTENT START ---\n{content}\n--- END FILE ---"
                    )
                except Exception as exc:
                    file_text_blocks.append(f"[Error reading file {el.name}: {exc}]")
    
    combined_user_content = message.content or ""
    if file_text_blocks:
        combined_user_content += "\n\n" + "\n\n".join(file_text_blocks)

    # 2. Run the internal SRL Chain (Routing -> Diagnosis -> Generation -> Check)
    ai_text = ""
    prefix = ""
    checkpoint = None
    decision = None
    route: dict[str, Any] = {
        "phase": current_phase,
        "strategy": "NONE",
        "confidence": 0.0,
    }

    async with cl.Step(name="Tutor is thinking...") as step:
        if tutor_type == "SRL Tutor":

            # ----------------------------------------------------------------
            # SPEED: run the router concurrently with any remaining I/O.
            # File reading already happened above; here we fire the router
            # immediately and run checkpoint_and_decide the moment it resolves.
            # This shaves the router's round-trip off the critical path when
            # there is nothing else to await — and keeps the pattern ready for
            # when you do have parallel work (e.g. fetching a resource).
            # ----------------------------------------------------------------
            route_task = asyncio.create_task(
                route_message(client, combined_user_content, llm_history, current_phase)
            )
            route = await route_task

            predicted_phase = cast(str, route.get("phase") or current_phase)
            confidence = float(route.get("confidence", 0.0) or 0.0)
            new_phase = update_phase(current_phase, predicted_phase, confidence)

            cl.user_session.set("current_phase", new_phase)
            route["phase"] = new_phase

            checkpoint, decision = await checkpoint_and_decide(
                client, route, llm_history, combined_user_content
            )

            logger.info("=" * 40)
            logger.info(f"SRL SESSION: {session_id} | USER: {student_id}")
            logger.info(f"PHASE: {route['phase']} (Confidence: {route.get('confidence')})")
            logger.info(
                f"CHECKPOINT: "
                f"Kind={checkpoint.request_kind} | "
                f"Stage={checkpoint.task_stage} | "
                f"Progress={checkpoint.progress_state} | "
                f"Attempt={checkpoint.has_attempt} | "
                f"Gap={checkpoint.context_gap}"
            )
            logger.info(
                f"LEARNER: "
                f"Expertise={checkpoint.expertise_level} | "
                f"Frustration={checkpoint.frustration_level} | "
                f"SRL={checkpoint.srl_focus} | "
                f"ImplAllowed={checkpoint.implementation_allowed}"
            )
            logger.info(
                f"DECISION: Level={decision.support_level} | "
                f"CanShowCode={decision.can_show_code}"
            )
            logger.info("=" * 40)

            # Generate tutor reply (uses native multi-turn history internally)
            ai_text = await generate_full_reply(
                client, route, checkpoint, decision, llm_history, combined_user_content
            )

            # ----------------------------------------------------------------
            # CONDITIONAL SAFETY CHECK
            # check_reply now skips the LLM call for low-risk support levels
            # (CLARIFY, QUESTION, HINT, REFLECT) and returns a pre-approved
            # CheckResult with was_skipped=True.  This removes one sequential
            # round-trip on the majority of turns.
            # ----------------------------------------------------------------
            check = await check_reply(
                client, route, checkpoint, decision, ai_text, llm_history, combined_user_content
            )

            if check.was_skipped:
                logger.info(
                    f"SAFETY CHECK: skipped (support_level={decision.support_level})"
                )
            else:
                logger.info(
                    f"SAFETY CHECK: is_safe={check.is_safe} | "
                    f"leaks_solution={check.leaks_solution} | "
                    f"reason={check.reason}"
                )

            if not check.is_safe or check.leaks_solution:
                logger.info(f"SAFETY TRIGGERED: {check.reason}. Rewriting...")
                ai_text = await rewrite_reply(
                    client, route, checkpoint, decision, ai_text, check,
                    llm_history, combined_user_content
                )
                prefix = "*(Self-Correction)*: "

        else:
            # Standard Tutor logic — unchanged
            system_prompt = build_system_prompt(tutor_type, route)
            ai_text = await run_tutor(client, system_prompt, llm_history)

    # Final out
    msg = cl.Message(content="")
    
    if prefix:
        await msg.stream_token(prefix)
        
    for chunk in ai_text.split(" "):
        await msg.stream_token(chunk + " ")
        await asyncio.sleep(0.01)  
    
    await msg.send()
    
    await step.remove()

    # 4. Update History
    llm_history.append({"role": "user", "content": combined_user_content, "timestamp": datetime.now().isoformat()})
    
    history_entry: dict[str, Any] = {
        "role": "assistant",
        "content": ai_text,
        "timestamp": datetime.now().isoformat(),
        "route": route,
        "diagnosis": checkpoint.__dict__ if checkpoint else None,
        "decision": decision.__dict__ if decision else None,
        "check": check.__dict__ if check else None,
        "draft_reply": ai_text,
    }
        
    llm_history.append(history_entry)
    cl.user_session.set("llm_history", llm_history)
    maybe_save(session_id, student_id, tutor_type, llm_history)

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
