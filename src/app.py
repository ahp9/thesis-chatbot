import chainlit as cl
from openai import AsyncOpenAI
import os
from pathlib import Path

from lib.transcript import save_conversation

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", "/prompts/ai_base_control.txt")
TRANSCRIPT_DIR = Path(os.getenv("TRANSCRIPT_DIR", "./transcripts"))
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

@cl.on_chat_start
async def start():
    # Load your prompt from your shared folder!
    # await cl.Message(
    #     content="Type END anytime to save and exit.",
    #     actions=[
    #         cl.Action(name="end_chat", value="end", label="End Chat")
    #     ]
    # ).send()
    
    thread_id = cl.user_session.get("thread_id")
    if not thread_id:
        thread_id = cl.context.session.id
        cl.user_session.set("thread_id", thread_id)
    
    # with open("../prompts/ai_base_control.txt", "r") as f:
    #     system_prompt = f.read()
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    
    cl.user_session.set("message_history", [{"role": "system", "content": system_prompt}])
    
    await cl.Message(content="Hello! I am your SRL Tutor. What are we learning today?").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("message_history")
    history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    await msg.send()
    
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
        stream=True
    )

    full_text = ""
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            if token:
                full_text += token
                await msg.stream_token(token)

    history.append({"role": "assistant", "content": full_text})
    cl.user_session.set("message_history", history)

@cl.on_chat_end
async def on_chat_end():
    # Called when user starts a new chat or the session ends
    thread_id = cl.user_session.get("thread_id") or cl.context.session.id
    history = cl.user_session.get("message_history") or []

    # Save full conversation once
    save_conversation(thread_id, history, TRANSCRIPT_DIR)

    # Optional: clear session state (not strictly necessary)
    cl.user_session.set("message_history", [])
    cl.user_session.set("thread_id", None)
    
# @cl.on_action_callback("end_chat")
# async def end_chat(action: cl.Action):
#     thread_id = cl.user_session.get("thread_id") or cl.context.session.id
#     history = cl.user_session.get("message_history") or []

#     save_conversation(thread_id, history, TRANSCRIPT_DIR)

#     await cl.Message(content="✅ Conversation saved. You can start a new chat anytime!").send()