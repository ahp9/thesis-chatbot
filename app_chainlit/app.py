import chainlit as cl
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

load_dotenv() # Loads your sk-... key from .env
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@cl.on_chat_start
async def start():
    # Load your prompt from your shared folder!
    with open("../prompts/ai_base_control.txt", "r") as f:
        system_prompt = f.read()
    
    cl.user_session.set("message_history", [{"role": "system", "content": system_prompt}])
    await cl.Message(content="Hello! I am your SRL Tutor. What are we learning today?").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("message_history")
    history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    
    stream = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history,
        stream=True
    )

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await msg.stream_token(token)

    history.append({"role": "assistant", "content": msg.content})
    await msg.send()
