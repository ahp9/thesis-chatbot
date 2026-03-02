import json
from typing import Any, Dict
from services.prompt_loader import load_prompt

ROUTER_MODEL = "gpt-4o-mini"

async def route_message(client, user_message: str) -> Dict[str, Any]:
    router_system = load_prompt("base/router_system_prompt.txt")

    resp = await client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": router_system},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )

    content = resp.choices[0].message.content or ""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Fallback if model returns non-JSON
        data = {
            "phase": "PERFORMANCE",
            "strategy": "NONE",
            "confidence": 0.2,
            "signals": ["router_json_parse_failed"],
            "one_optional_question": ""
        }

    # Normalize
    data["phase"] = (data.get("phase") or "PERFORMANCE").upper()
    data["strategy"] = (data.get("strategy") or "NONE").upper()
    return data