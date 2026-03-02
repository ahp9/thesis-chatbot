import os
from openai import AsyncOpenAI

def get_client() -> AsyncOpenAI:
    api_key = os.environ["OPENAI_API_KEY"]
    return AsyncOpenAI(api_key=api_key)