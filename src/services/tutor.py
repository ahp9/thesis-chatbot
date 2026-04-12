from typing import Any
from pathlib import Path
import logging

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from services.llm_client import get_client

client = get_client()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

BASIC_MODEL = "gpt-4.1-mini"
_BASIC_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "base" / "ai_base_control.txt"
)

def _load_basic_system_prompt() -> str:
    try:
        return _BASIC_SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning("Could not load basic system prompt (%s) — using inline fallback.", exc)
        return (
            "You are a helpful tutor assistant. Answer questions directly, "
            "write and explain code, solve problems, and support students with "
            "coursework or projects. Be clear, thorough, and solution-oriented."
        )


async def _run_basic_tutor(
    llm_history: list[dict[str, Any]],
    user_message: str,
) -> str:
    system_prompt = _load_basic_system_prompt()

    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content=system_prompt)
    ]

    for item in llm_history:
        role = item.get("role", "")
        if role not in ("user", "assistant"):
            continue

        content = str(item.get("content") or "").strip()
        if not content:
            continue

        if "--- FILE:" in content:
            content = content.split("--- FILE:")[0].strip() or "[previous file upload]"

        if role == "user":
            messages.append(ChatCompletionUserMessageParam(role="user", content=content))
        else:
            messages.append(
                ChatCompletionAssistantMessageParam(role="assistant", content=content)
            )

    messages.append(ChatCompletionUserMessageParam(role="user", content=user_message))

    resp = await client.chat.completions.create(
        model=BASIC_MODEL,
        messages=messages,
        temperature=0.7,
    )
    return resp.choices[0].message.content or ""