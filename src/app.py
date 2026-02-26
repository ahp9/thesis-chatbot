import os
import json
import sqlite3
import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict
from datetime import datetime
from typing import Optional
from openai import AsyncOpenAI

from utils.logger import save_conversation

# --- FIX 1 & 2: DATABASE ADAPTERS ---
# This ensures lists and dicts save correctly in SQLite without Pydantic errors
sqlite3.register_adapter(list, lambda lst: json.dumps(lst))
sqlite3.register_adapter(dict, lambda dct: json.dumps(dct))

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

MOCK_USERS = {
    "student1@research.local": "password123",
    "student2@research.local": "study2026",
    "student3@research.local": "research_mode"
}

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    if username in MOCK_USERS and MOCK_USERS[username] == password:
        return cl.User(identifier=username, name=username.split("@")[0])
    return None

@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo="sqlite+aiosqlite:///./chainlit.db")

@cl.on_chat_start
async def start():
    user = cl.user_session.get("user")
    student_id = user.identifier.split("@")[0]
    
    chat_profile = cl.user_session.get("chat_profile")
    
    # Store information
    cl.user_session.set("user_id", student_id)
    cl.user_session.set("tutor_type", chat_profile)
    cl.user_session.set("message_history", [])
    thread_id = cl.context.session.thread_id
    cl.user_session.set("session_id", thread_id)
    
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
    history = []
    for step in steps:
        role = "assistant" if step.get("type") == "assistant_message" else "user"
        content = step.get("output") or step.get("input") or ""
        history.append({"role": role, "content": content})
    
    # Restore the session variables
    cl.user_session.set("message_history", history)


    

    
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
        cl.ChatProfile(
            name="Strict Socratic",
            markdown_description="Only asks guiding questions.",
        ),
    ]

@cl.on_message
async def main(message: cl.Message):
    # Fetch the message history and phase from the session and AI type
    history = cl.user_session.get("message_history")
    tutor_type = cl.user_session.get("tutor_type") 
    session_id = cl.user_session.get("session_id") 
    student_id = cl.user_session.get("user_id")
    
    history.append({"role": "user", "content": message.content, "timestamp": datetime.now().isoformat()})

    # Create a system prompt based on the tutor type and phase
    if tutor_type == "SRL Tutor":
        system_prompt = f"You are a Self-Regulated Learning (SRL) tutor. Your role is to guide students through the phases of forethought, performance, and self-reflection. Currently, Provide appropriate prompts and feedback based on the student's input and the current phase."
    elif tutor_type == "Basic Tutor":
        system_prompt = f"You are a helpful AI assistant. Give direct answers and code when asked."
    else:
        system_prompt = f"You are a strict Socratic tutor. Always respond with questions that guide the student to find the answer themselves. Do not provide direct answers or code."

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            *history
        ]
    )
    
    ai_text = response.choices[0].message.content

    # Save the AI response to history
    history.append({"role": "assistant", "content": ai_text, "timestamp": datetime.now().isoformat(), "system_prompt": system_prompt})
    cl.user_session.set("message_history", history)
    
    if len(history) > 0:
        save_conversation(
            session_id=session_id,
            user_id=student_id,
            tutor_type=cl.user_session.get("tutor_type"),
            history=history
        )
    
    await cl.Message(content=ai_text).send()
    
@cl.on_chat_end
async def end():
    history = cl.user_session.get("message_history")
    if history: # Only save if there was a conversation
        save_conversation(
            session_id=cl.user_session.get("session_id"),
            user_id=cl.user_session.get("user_id"),
            tutor_type=cl.user_session.get("tutor_type"),
            history=history
        )

