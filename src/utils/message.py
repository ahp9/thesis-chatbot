import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import chainlit as cl
from chainlit.element import File

from utils.file import read_uploaded_file

MAX_CHARS = 80_000

logger = logging.getLogger(__name__)


async def persist_uploaded_elements_for_message(
    incoming_message: cl.Message,
    assistant_message_id: str,
) -> list[dict[str, str]]:
    """Attach uploaded files from the user message to the assistant reply in Chainlit storage."""
    persisted_info: list[dict[str, str]] = []
    if not incoming_message.elements:
        return persisted_info

    outgoing_elements = []
    for el in incoming_message.elements:
        if not isinstance(el, File):
            continue

        path_str = el.path
        if path_str is None:
            continue

        file_name = el.name or Path(path_str).name
        mime = el.mime or "application/octet-stream"

        try:
            outgoing_elements.append(
                cl.File(
                    name=file_name,
                    path=path_str,
                    mime=mime,
                    for_id=assistant_message_id,
                )
            )
            persisted_info.append(
                {
                    "name": file_name,
                    "mime": mime,
                    "path": path_str,
                }
            )
        except Exception as exc:
            logger.warning(
                "Failed to prepare file element %s: %s",
                getattr(el, "name", "?"),
                exc,
            )

    if outgoing_elements:
        await cl.Message(content="", elements=outgoing_elements).send()

    return persisted_info


def merge_db_steps_with_transcript(
    steps: Sequence[Mapping[str, Any]],
    saved_history: list[dict],
) -> list[dict]:
    """Merge Chainlit DB steps (authoritative for turn order) with saved SRL metadata (authoritative for rich fields).

    DB turns are deduplicated then matched positionally to saved user/assistant turns.
    """
    # Extract (role, content) pairs from DB steps
    raw: list[tuple[str, str]] = []
    for step in steps:
        role = "assistant" if step.get("type") == "assistant_message" else "user"
        content = step.get("output") or step.get("input") or ""
        if not content or content == "{}":
            continue
        raw.append((role, content))

    # Deduplicate consecutive identical (role, content) pairs
    deduped: list[tuple[str, str]] = []
    for role, content in raw:
        if deduped and deduped[-1] == (role, content):
            continue
        deduped.append((role, content))

    # Match each deduped turn to saved metadata by position
    saved_user: list[dict] = [t for t in saved_history if t.get("role") == "user"]
    saved_asst: list[dict] = [t for t in saved_history if t.get("role") == "assistant"]
    user_idx = 0
    asst_idx = 0

    merged: list[dict] = []
    for role, content in deduped:
        if role == "user":
            saved_turn = saved_user[user_idx] if user_idx < len(saved_user) else {}
            user_idx += 1
        else:
            saved_turn = saved_asst[asst_idx] if asst_idx < len(saved_asst) else {}
            asst_idx += 1

        if saved_turn:
            entry = dict(saved_turn)
            entry["content"] = content  # DB content is authoritative
        else:
            entry = {"role": role, "content": content}

        merged.append(entry)

    return merged


def build_combined_user_content(message: cl.Message) -> tuple[str, list[dict[str, str]]]:
    """Combine the user's text with any uploaded file contents into a single string."""
    file_text_blocks: list[str] = []
    uploaded_files_meta: list[dict[str, str]] = []

    if message.elements:
        for el in message.elements:
            if not isinstance(el, File):
                continue
            path_str = el.path
            if path_str is None:
                continue
            file_name = el.name or Path(path_str).name
            mime = el.mime or "application/octet-stream"
            try:
                content = read_uploaded_file(el)
                content = content[:MAX_CHARS]
                uploaded_files_meta.append(
                    {
                        "name": file_name,
                        "mime": mime,
                        "path": path_str,
                    }
                )
                file_text_blocks.append(
                    f"--- FILE: {file_name} ({mime}) ---\n"
                    f"CONTENT START\n"
                    f"{content}\n"
                    f"CONTENT END\n"
                    f"--- END FILE ---"
                )
            except Exception as exc:
                file_text_blocks.append(
                    f"[Error reading file {getattr(el, 'name', 'unknown file')}: {exc}]"
                )

    combined_user_content = message.content or ""
    if file_text_blocks:
        combined_user_content += "\n\n" + "\n\n".join(file_text_blocks)

    return combined_user_content, uploaded_files_meta
