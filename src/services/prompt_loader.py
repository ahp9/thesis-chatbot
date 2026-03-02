from pathlib import Path
from functools import lru_cache

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"

@lru_cache(maxsize=128)
def load_prompt(rel_path: str) -> str:
    path = PROMPTS_DIR / rel_path
    return path.read_text(encoding="utf-8")