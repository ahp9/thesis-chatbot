import chainlit as cl
from openai import AsyncOpenAI
import os
from chainlit.input_widget import Select

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 
client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", "/prompts/ai_base_control.txt")
# TRANSCRIPT_DIR = Path(os.getenv("TRANSCRIPT_DIR", "./transcripts"))
# TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

@cl.on_chat_start
async def start():
    chat_profile = cl.user_session.get("chat_profile")
    
    # Store it in our session so main() can find it
    cl.user_session.set("tutor_type", chat_profile)
    
    cl.user_session.set("phase", "forethought")
    cl.user_session.set("message_history", [])

    
@cl.on_settings_update
async def update_settings(settings):
    new_tutor = settings["tutor_type"]
    cl.user_session.set("tutor_type", new_tutor)
    
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
    history = cl.user_session.get("message_history")
    phase = cl.user_session.get("phase")
    tutor_type = cl.user_session.get("tutor_type")  
    
    history.append({"role": "user", "content": message.content})

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

    # 6. Simple Phase Switch (Example: Move to performance after the first goal is set)
    if phase == "forethought":
        cl.user_session.set("phase", "performance")

    # 7. Save and Send
    history.append({"role": "assistant", "content": ai_text})
    cl.user_session.set("history", history)
    
    await cl.Message(content=ai_text).send()

