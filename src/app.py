import json
import logging
import sqlite3
import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from datetime import datetime
from typing import Any, Dict, Optional

from services.llm_client import get_client
from services.router import route_message, update_phase
from services.tutor import build_system_prompt, run_tutor
from utils.logger import save_conversation

# This ensures lists and dicts save correctly in SQLite without Pydantic errors
sqlite3.register_adapter(list, lambda lst: json.dumps(lst))
sqlite3.register_adapter(dict, lambda dct: json.dumps(dct))

client = get_client()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MOCK_USERS = {
    "student1@research.local": "password123",
    "student2@research.local": "study2026",
    "student3@research.local": "research_mode",
    "working_mode@admin.local": "working_mode"
}

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    if username in MOCK_USERS and MOCK_USERS[username] == password:
        return cl.User(identifier=username, name=username.split("@")[0])
    return None

@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo="sqlite+aiosqlite:///./chainlit.db")

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="SRL Tutor",
            markdown_description="Focuses on planning, doing, and reflecting.",
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
    
    # Store information
    cl.user_session.set("user_id", student_id)
    cl.user_session.set("tutor_type", tutor_type)
    cl.user_session.set("session_id", thread_id)
    
    cl.user_session.set("llm_history", [])
    
    cl.user_session.set("current_phase", "FORETHOUGHT")
    
    
@cl.on_chat_resume
async def on_chat_resume(thread : ThreadDict):
    metadata = thread.get("metadata", {})
    
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
            
    student_id = metadata.get("user_id", "Unknown")
    tutor_type = metadata.get("tutor_type", "SRL Tutor")

    cl.user_session.set("user_id", student_id)
    cl.user_session.set("tutor_type", tutor_type)
    cl.user_session.set("session_id", thread.get("id"))
    
    # This restores the message history from the database
    steps = thread.get("steps", [])
    
    # Reconstruct the message history for your OpenAI call
    llm_history = []
    for step in steps:
        role = "assistant" if step.get("type") == "assistant_message" else "user"
        content = step.get("output") or step.get("input") or ""
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
    # Fetch the message history and phase from the session and AI type
    tutor_type = cl.user_session.get("tutor_type") 
    session_id = cl.user_session.get("session_id") 
    student_id = cl.user_session.get("user_id")
    llm_history = cl.user_session.get("llm_history")
    current_phase = cl.user_session.get("current_phase")
    
    llm_history.append({"role": "user", "content": message.content, "timestamp": datetime.now().isoformat()})

    # Background router (only for SRL Tutor)
    route: Dict[str, Any] = {"phase": current_phase, "strategy": "NONE", "confidence": 0.0}
    if tutor_type == "SRL Tutor":
        route = await route_message(client, message.content, llm_history, current_phase)
        
        predicted_phase = route.get("phase", current_phase)
        confidence = route.get("confidence", 0.0)
        new_phase = update_phase(current_phase, predicted_phase, confidence)
        cl.user_session.set("current_phase", new_phase)
        
        route["phase"] = new_phase

    cl.user_session.set("last_route", route)
    
    system_prompt = build_system_prompt(tutor_type, route)  
    
    logger.info(f"System Prompt for session {session_id}: {system_prompt[:200]!r}") 
    logger.info(f" DEBUG Route={route}") 
    ai_text = await run_tutor(client, system_prompt, llm_history)

    # Save the AI response to history
    llm_history.append({"role": "assistant", "content": ai_text, "timestamp": datetime.now().isoformat(), "system_prompt": system_prompt})
    cl.user_session.set("llm_history", llm_history)
    
    maybe_save(session_id=session_id, student_id=student_id, tutor_type=tutor_type, llm_history=llm_history)
    
    await cl.Message(content=ai_text).send()
    
@cl.on_chat_end
async def end():
    session_id = cl.user_session.get("session_id")
    student_id = cl.user_session.get("user_id")
    tutor_type = cl.user_session.get("tutor_type") or "SRL Tutor"
    llm_history = cl.user_session.get("llm_history")
    maybe_save(session_id=session_id, student_id=student_id, tutor_type=tutor_type, llm_history=llm_history)

