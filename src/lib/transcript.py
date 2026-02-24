import json
from datetime import datetime
from pathlib import Path

def _now_iso():
    return datetime.utcnow().isoformat() + "Z"

def _safe_filename(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in ("-", "_"))

def save_conversation(thread_id: str, history: list[dict], TRANSCRIPT_DIR: Path = Path("transcripts")):
    # Save the whole conversation as one JSON file
    payload = {
        "thread_id": thread_id,
        "saved_at": _now_iso(),
        "messages": history,
    }

    path = TRANSCRIPT_DIR / f"{_safe_filename(thread_id)}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)