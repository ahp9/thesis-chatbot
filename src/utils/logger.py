import json
import os
from datetime import datetime

LOG_FILE = "transcripts"

def get_log_filename(user_id, session_id):
    return os.path.join(LOG_FILE, f"user_{user_id}_session_{session_id}.jsonl")

def save_conversation(session_id, user_id, tutor_type, history):
    
    os.makedirs(LOG_FILE, exist_ok=True)
    filename = get_log_filename(user_id, session_id)
    
    data = {
        "metadata": {
            "session_id": session_id,
            "user_id": user_id,
            "tutor_type": tutor_type,
            "timestamp": datetime.now().isoformat()
        },
        "history": history
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)