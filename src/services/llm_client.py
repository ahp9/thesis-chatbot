import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()  # loads .env from project root by default


def get_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Set it in your environment or .env file."
        )
    return AsyncOpenAI(api_key=api_key)
