import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiofiles
import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.element import File
from chainlit.types import ThreadDict

from chainlit.data.storage_clients.base import BaseStorageClient

from lib.enums import Phase, TutorMode
from services.llm_client import get_client
from services.orchestrator import Orchestrator
from services.tutor import _run_basic_tutor
from utils.file import read_uploaded_file
from utils.logger import save_conversation

sqlite3.register_adapter(list, lambda lst: json.dumps(lst))
sqlite3.register_adapter(dict, lambda dct: json.dumps(dct))

client = get_client()
orchestrator = Orchestrator(client)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

MOCK_USERS = {
    "student1@research.local": "password123",
    "student2@research.local": "study2026",
    "student3@research.local": "research_mode",
    "working_mode@admin.local": "working_mode",
    "user_1@usability_test_1.local": "usability1",
    "user_2@usability_test_2.local": "usability2",
    "user_3@usability_test_3.local": "usability3",
    "user_3@usability_test_4.local": "usability4",
    "user_5@usability_test_5.local": "usability5",
    "user_1_2@usability_test_1_2.local": "usability1_2",
    "user_2_2@usability_test_2_2.local": "usability2_2",
    "user_3_2@usability_test_3_2.local": "usability3_2",
    "user_4_2@usability_test_4.local": "usability4_2",
    "user_5_2@usability_test_5_2.local": "usability5_2",
    
    "pilot_1@experiment.local": "pilot1",
    
    "experiment_1@experiement.local": "experiment1",
    "experiment_2@experiement.local": "experiment2",
    "experiment_3@experiement.local": "experiment3",
    "experiment_4@experiement.local": "experiment4",
    "experiment_5@experiement.local": "experiment5",
    "experiment_6@experiement.local": "experiment6",
    "experiment_7@experiement.local": "experiment7",
    "experiment_8@experiement.local": "experiment8",
    "experiment_9@experiement.local": "experiment9",
    "experiment_10@experiement.local": "experiment10",
    "experiment_11@experiement.local": "experiment11",
    "experiment_12@experiement.local": "experiment12",
    "experiment_13@experiement.local": "experiment13",
    "experiment_14@experiement.local": "experiment14",
    "experiment_15@experiement.local": "experiment15",
    "experiment_16@experiement.local": "experiment16",
    "experiment_17@experiement.local": "experiment17",
    "experiment_18@experiement.local": "experiment18",
    "experiment_19@experiement.local": "experiment19",
    "experiment_20@experiement.local": "experiment20",
}

MAX_CHARS = 80_000
LOG_FILE = "transcripts"
UPLOAD_ROOT = Path("./uploaded_files")

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class LocalStorageClient(BaseStorageClient):
    """
    Minimal local filesystem storage provider for Chainlit elements.
    """

    def __init__(self, root: str | Path = "./uploaded_files"):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, object_key: str) -> Path:
        safe_key = object_key.lstrip("/").replace("..", "_")
        path = self.root / safe_key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    async def close(self):
        pass

    async def upload_file(
        self,
        object_key: str,
        data: bytes | str,
        mime: str = "application/octet-stream",
        overwrite: bool = True,
        content_disposition: str | None = None,
    ) -> dict[str, Any]:
        path = self._path_for_key(object_key)
        if path.exists() and not overwrite:
            return {"url": path.resolve().as_uri(), "object_key": object_key}

        write_data = data.encode("utf-8") if isinstance(data, str) else data

        async with aiofiles.open(path, "wb") as f:
            await f.write(write_data)

        return {
            "url": path.resolve().as_uri(),
            "object_key": object_key,
            "content_disposition": content_disposition,
        }

    async def delete_file(self, object_key: str):
        path = self._path_for_key(object_key)
        if path.exists():
            path.unlink()

    async def get_read_url(self, object_key: str):
        path = self._path_for_key(object_key)
        if not path.exists():
            raise FileNotFoundError(f"Stored file not found for object_key={object_key}")
        return path.resolve().as_uri()


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def get_log_filename(user_id, session_id):
    os.makedirs(LOG_FILE, exist_ok=True)
    return os.path.join(LOG_FILE, f"user_{user_id}_session_{session_id}.json")


def load_conversation(user_id, session_id):
    filename = get_log_filename(user_id, session_id)
    if not os.path.exists(filename):
        return None
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Could not load transcript %s: %s", filename, exc)
        return None


def maybe_save(session_id: str, student_id: str, tutor_type: str, llm_history):
    if student_id != "working_mode" and llm_history:
        logger.info("Saving transcript...")
        save_conversation(
            session_id=session_id,
            user_id=student_id,
            tutor_type=tutor_type,
            history=llm_history,
        )


def _coerce_phase(value: Any) -> Phase:
    try:
        return Phase(str(value or "FORETHOUGHT").upper())
    except ValueError:
        return Phase.FORETHOUGHT


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

async def _persist_uploaded_elements_for_message(
    incoming_message: cl.Message,
    assistant_message_id: str,
) -> list[dict[str, str]]:
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


def _build_combined_user_content(message: cl.Message) -> tuple[str, list[dict[str, str]]]:
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


# ---------------------------------------------------------------------------
# Chainlit setup
# ---------------------------------------------------------------------------

@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> Optional[cl.User]:
    if username in MOCK_USERS and MOCK_USERS[username] == password:
        return cl.User(
            identifier=username,
            metadata={
                "display_name": username.split("@")[0],
                "provider": "credentials",
            },
        )
    return None


@cl.data_layer
def get_data_layer():
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    return SQLAlchemyDataLayer(
        conninfo="sqlite+aiosqlite:///./chainlit.db",
        storage_provider=LocalStorageClient(UPLOAD_ROOT),
    )


@cl.set_chat_profiles
async def chat_profile(current_user: cl.User | None) -> list[cl.ChatProfile]:
    return [
        cl.ChatProfile(
            name=TutorMode.SRL.value,
            markdown_description="Phase-aware chained tutoring with pushback.",
        ),
        cl.ChatProfile(
            name=TutorMode.BASIC.value,
            markdown_description="Direct answers and code support.",
        ),
    ]


@cl.on_chat_start
async def start():
    user = cl.user_session.get("user")
    student_id = user.identifier.split("@")[0]
    tutor_type = cl.user_session.get("chat_profile") or TutorMode.SRL.value
    thread_id = cl.context.session.thread_id

    cl.user_session.set("user_id", student_id)
    cl.user_session.set("tutor_type", tutor_type)
    cl.user_session.set("session_id", thread_id)
    cl.user_session.set("llm_history", [])
    cl.user_session.set("current_phase", Phase.FORETHOUGHT.value)

    logger.info("Chat started | user=%s | thread=%s | mode=%s", student_id, thread_id, tutor_type)


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    raw_metadata = thread.get("metadata", {})
    metadata: dict[str, Any]

    if isinstance(raw_metadata, str):
        try:
            metadata = json.loads(raw_metadata)
        except json.JSONDecodeError:
            metadata = {}
    elif isinstance(raw_metadata, dict):
        metadata = raw_metadata
    else:
        metadata = {}

    student_id = metadata.get("user_id", "Unknown")
    tutor_type = metadata.get("tutor_type", TutorMode.SRL.value)
    session_id = thread.get("id")

    cl.user_session.set("user_id", student_id)
    cl.user_session.set("tutor_type", tutor_type)
    cl.user_session.set("session_id", session_id)

    saved = load_conversation(student_id, session_id)

    if saved and isinstance(saved.get("history"), list):
        llm_history = saved["history"]
        cl.user_session.set("llm_history", llm_history)

        current_phase = metadata.get("current_phase")
        if not current_phase:
            current_phase = Phase.FORETHOUGHT.value
            for item in reversed(llm_history):
                if item.get("role") == "assistant" and isinstance(item.get("route"), dict):
                    current_phase = item["route"].get("phase", Phase.FORETHOUGHT.value)
                    break

        cl.user_session.set("current_phase", current_phase)
        logger.info("Resumed saved transcript for %s, session %s", student_id, session_id)
        return

    steps = thread.get("steps", [])
    llm_history = []

    for step in steps:
        role = "assistant" if step.get("type") == "assistant_message" else "user"
        content = step.get("output") or step.get("input") or ""
        if content and content != "{}":
            llm_history.append({"role": role, "content": content})

    cl.user_session.set("llm_history", llm_history)
    cl.user_session.set(
        "current_phase",
        metadata.get("current_phase", Phase.FORETHOUGHT.value),
    )

    logger.info("Resumed thread from Chainlit history | thread=%s | steps=%d", session_id, len(steps))


# ---------------------------------------------------------------------------
# Main message handler
# ---------------------------------------------------------------------------

@cl.on_message
async def main(message: cl.Message):
    tutor_type = str(cl.user_session.get("tutor_type") or TutorMode.SRL.value)
    session_id = cl.user_session.get("session_id")
    student_id = cl.user_session.get("user_id")
    llm_history: list[dict[str, Any]] = cl.user_session.get("llm_history") or []
    current_phase = _coerce_phase(cl.user_session.get("current_phase"))

    combined_user_content, uploaded_files_meta = _build_combined_user_content(message)

    ai_text = ""
    prefix = ""

    # SRL-mode metadata — only populated in SRL mode.
    route_for_history: dict[str, Any] = {
        "phase": current_phase.value,
        "strategy": "NONE",
        "confidence": 0.0,
        "signals": [],
    }
    checkpoint_for_history = None
    decision_for_history = None
    safety_for_history = None
    draft_reply_for_history = ""

    is_srl = tutor_type == TutorMode.SRL.value

    async with cl.Step(name="Thinking...") as step:

        if is_srl:
            # ----------------------------------------------------------------
            # SRL Tutor — full pipeline: route → classify → generate → safety
            # ----------------------------------------------------------------
            result = await orchestrator.handle_turn(
                user_message=combined_user_content,
                llm_history=llm_history,
                current_phase=current_phase,
            )

            ai_text = result.reply
            prefix = result.prefix
            draft_reply_for_history = result.draft_reply
            route_for_history = result.route.to_dict()
            checkpoint_for_history = result.control.checkpoint.to_dict()
            decision_for_history = result.control.decision.to_dict()
            safety_for_history = result.safety.to_dict()

            cl.user_session.set("current_phase", result.route.phase.value)

            logger.info("=" * 40)
            logger.info("SRL SESSION: %s | USER: %s", session_id, student_id)
            logger.info(
                "PHASE: %s (Confidence: %.2f) | SRL Signals: %s",
                result.route.phase.value,
                result.route.confidence,
                result.route.srl_signal,
            )
            logger.info(
                "CHECKPOINT: Kind=%s | Stage=%s | Progress=%s | Attempt=%s | Gap=%s | SRL=%s | SubtaskScope=%s",
                result.control.checkpoint.request_kind.value,
                result.control.checkpoint.task_stage.value,
                result.control.checkpoint.progress_state.value,
                result.control.checkpoint.has_attempt,
                result.control.checkpoint.context_gap.value,
                result.control.checkpoint.srl_focus.value,
                result.control.checkpoint.subtask_scope,
            )
            logger.info(
                "LEARNER: Expertise=%s | Frustration=%s | SRL=%s",
                result.control.checkpoint.expertise_level.value,
                result.control.checkpoint.frustration_level.value,
                result.control.checkpoint.srl_focus.value,
            )
            logger.info(
                "DECISION: Level=%s | Depth=%s | CanShowCode=%s",
                result.control.decision.support_level.value,
                result.control.decision.support_depth.value,
                result.control.decision.can_show_code,
            )
            logger.info(
                "SAFETY CHECK: skipped=%s | is_safe=%s | leaks_solution=%s | reason=%s",
                result.safety.was_skipped,
                result.safety.is_safe,
                result.safety.leaks_solution,
                result.safety.reason,
            )
            logger.info("=" * 40)

        else:
            # ----------------------------------------------------------------
            # Basic Tutor — single LLM call, no classification or safety chain
            # ----------------------------------------------------------------
            logger.info("BASIC SESSION: %s | USER: %s", session_id, student_id)
            ai_text = await _run_basic_tutor(
                llm_history,
                combined_user_content,
            )

    # -------------------------------------------------------------------------
    # Stream reply
    # -------------------------------------------------------------------------
    msg = cl.Message(content="")
    await msg.send()

    if prefix:
        await msg.stream_token(prefix)

    for chunk in ai_text.split(" "):
        await msg.stream_token(chunk + " ")
        await asyncio.sleep(0.01)

    await msg.update()
    await step.remove()

    # Persist uploaded files as message elements in Chainlit storage.
    persisted_files = await _persist_uploaded_elements_for_message(message, msg.id)

    # -------------------------------------------------------------------------
    # Update history
    # -------------------------------------------------------------------------
    llm_history.append(
        {
            "role": "user",
            "content": combined_user_content,
            "timestamp": datetime.now().isoformat(),
            "uploaded_files": uploaded_files_meta,
        }
    )

    history_entry: dict[str, Any] = {
        "role": "assistant",
        "content": ai_text,
        "timestamp": datetime.now().isoformat(),
        "draft_reply": draft_reply_for_history or ai_text,
        "persisted_files": persisted_files,
    }

    # Only attach SRL metadata when it was actually computed.
    if is_srl:
        history_entry["route"] = route_for_history
        history_entry["diagnosis"] = checkpoint_for_history
        history_entry["decision"] = decision_for_history
        history_entry["check"] = safety_for_history

    llm_history.append(history_entry)
    cl.user_session.set("llm_history", llm_history)
    maybe_save(session_id, student_id, tutor_type, llm_history)


@cl.on_chat_end
async def end():
    session_id = cl.user_session.get("session_id")
    student_id = cl.user_session.get("user_id")
    tutor_type = cl.user_session.get("tutor_type") or TutorMode.SRL.value
    llm_history = cl.user_session.get("llm_history")

    maybe_save(
        session_id=session_id,
        student_id=student_id,
        tutor_type=tutor_type,
        llm_history=llm_history,
    )