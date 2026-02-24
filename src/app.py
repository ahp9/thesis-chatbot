from datetime import datetime

import chainlit as cl
from openai import AsyncOpenAI
import os

from utils.logger import save_conversation

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

@cl.on_chat_start
async def start():
    
    res = await cl.AskUserMessage(content="Please enter the Tester ID or Name to begin:", timeout=60).send()
    
    
    if res:
        tester_id = res["output"]
        cl.user_session.set("user_id", res["output"])
    
        chat_profile = cl.user_session.get("chat_profile")
    
        # Store information
        cl.user_session.set("tutor_type", chat_profile)
        cl.user_session.set("phase", "forethought")
        cl.user_session.set("message_history", [])
        cl.user_session.set("session_id", cl.context.session.id)
        
        await cl.Message(content=f"Welcome, **{tester_id}**. How can i help you?").send()
    

    
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
    phase = cl.user_session.get("phase")
    tutor_type = cl.user_session.get("tutor_type") 
    session_id = cl.user_session.get("session_id") 
    student_id = cl.user_session.get("user_id")
    
    history.append({"role": "user", "content": message.content, "timestamp": datetime.now().isoformat()})

    # Create a system prompt based on the tutor type and phase
    if tutor_type == "SRL Tutor":
        system_prompt = f"You are a Self-Regulated Learning (SRL) tutor. Your role is to guide students through the phases of forethought, performance, and self-reflection. Currently, we are in the {phase} phase. Provide appropriate prompts and feedback based on the student's input and the current phase."
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

    # Change to next phase
    if phase == "forethought":
        cl.user_session.set("phase", "performance")

    # Save the AI response to history
    history.append({"role": "assistant", "content": ai_text, "timestamp": datetime.now().isoformat(), "system_prompt": system_prompt})
    cl.user_session.set("message_history", history)
    
    if len(history) > 0:
        save_conversation(
            session_id=session_id,
            user_id=student_id,
            tutor_type=cl.user_session.get("tutor_type"),
            phase=cl.user_session.get("phase"),
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
            phase=cl.user_session.get("phase"),
            history=history
        )

