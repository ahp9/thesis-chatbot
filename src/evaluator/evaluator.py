import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from services.llm_client import get_client
from services.router import route_message
from services.tutor import build_system_prompt, run_tutor
from evaluator.judge import JUDGE_SYSTEM, build_judge_user_prompt

ROOT = Path(__file__).resolve().parents[1]
RUBRICS_DIR = ROOT / "evaluator" / "rubrics"
PROMPTS_DIR = ROOT / "evaluator" / "prompts"
REPORTS_DIR = ROOT / "evaluator" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

JUDGE_MODEL = "gpt-4o-mini"
JUDGE_TEMP = 0.0

BASE_SRL_VERSION = "v3"
SRL_PHASE_VERSION = "v2"

PHASE_ORDER = ["FORETHOUGHT", "PERFORMANCE", "REFLECTION"]

# Timeouts (seconds) — tune as needed
ROUTE_TIMEOUT_S = 45
TUTOR_TIMEOUT_S = 90
JUDGE_TIMEOUT_S = 90


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_suite(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    # Helpful debug if JSON is invalid / empty
    if not raw.strip():
        raise ValueError(f"Prompt suite file is empty: {path}")
    return json.loads(raw)


def require_file(path: Path, kind: str):
    if not path.exists():
        raise FileNotFoundError(f"{kind} file not found: {path}")


def format_transcript(turns: List[Dict[str, str]], assistant_turns: List[str]) -> str:
    out = []
    a_i = 0
    for t in turns:
        if t["role"] == "user":
            out.append(f"USER: {t['content']}")
            if a_i < len(assistant_turns):
                out.append(f"ASSISTANT: {assistant_turns[a_i]}")
                a_i += 1
        else:
            out.append(f"{t['role'].upper()}: {t['content']}")
    return "\n".join(out)


async def _with_timeout(coro, seconds: int, label: str):
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError as e:
        raise TimeoutError(f"Timed out during {label} after {seconds}s") from e


def _update_phase(current_phase: str, predicted_phase: str, confidence: float) -> str:
    current_phase = (current_phase or "FORETHOUGHT").upper()
    predicted_phase = (predicted_phase or current_phase).upper()
    confidence = float(confidence or 0.0)

    if current_phase not in PHASE_ORDER:
        current_phase = "FORETHOUGHT"
    if predicted_phase not in PHASE_ORDER:
        predicted_phase = current_phase

    # If router is unsure, don't change state
    if confidence < 0.60:
        return current_phase

    # Move forward only
    if PHASE_ORDER.index(predicted_phase) >= PHASE_ORDER.index(current_phase):
        return predicted_phase

    return current_phase


async def run_suite(
    suite_file: str,
    rubric_file: str,
    tutor_type: str = "SRL Tutor",
    max_cases: Optional[int] = None,
    verbose: bool = True,
):
    client = get_client()

    rubric_path = RUBRICS_DIR / rubric_file
    suite_path = PROMPTS_DIR / suite_file
    require_file(rubric_path, "Rubric")
    require_file(suite_path, "Prompt suite")

    rubric = load_yaml(rubric_path)
    suite = load_suite(suite_path)

    if max_cases is not None:
        suite = suite[:max_cases]

    if verbose:
        print(f"[EVAL] Suite: {suite_path} ({len(suite)} cases)")
        print(f"[EVAL] Rubric: {rubric_path}")
        print(f"[EVAL] Tutor type: {tutor_type}")
        print(
            f"[EVAL] Timeouts: route={ROUTE_TIMEOUT_S}s tutor={TUTOR_TIMEOUT_S}s judge={JUDGE_TIMEOUT_S}s"
        )

    results = []

    for idx, case in enumerate(suite, start=1):
        case_id = case.get("id", f"case_{idx}")
        turns = case["turns"]
        llm_history: List[Dict[str, str]] = []
        assistant_outputs: List[str] = []

        current_phase = case.get("start_phase", "FORETHOUGHT").upper()

        if verbose:
            print(
                f"\n[EVAL] Case {idx}/{len(suite)}: {case_id} (start_phase={current_phase}, turns={len(turns)})"
            )

        for t_i, turn in enumerate(turns, start=1):
            if turn["role"] != "user":
                continue

            user_msg = turn["content"]
            llm_history.append({"role": "user", "content": user_msg})

            route = {"phase": current_phase, "strategy": "NONE", "confidence": 0.0}

            if tutor_type == "SRL Tutor":
                if verbose:
                    print(f"[EVAL]   Turn {t_i}: routing...")
                route = await _with_timeout(
                    route_message(client, user_msg, llm_history, current_phase),
                    ROUTE_TIMEOUT_S,
                    label=f"routing (case={case_id}, turn={t_i})",
                )

                predicted_phase = route.get("phase", current_phase)
                conf = float(route.get("confidence", 0.0))
                current_phase = _update_phase(current_phase, predicted_phase, conf)
                route["phase"] = current_phase

                if verbose:
                    print(
                        f"[EVAL]   Turn {t_i}: route.phase={route.get('phase')} conf={conf:.2f} strategy={route.get('strategy', 'NONE')}"
                    )

            system_prompt = build_system_prompt(tutor_type, route)

            if verbose:
                print(f"[EVAL]   Turn {t_i}: tutoring...")
            assistant_text = await _with_timeout(
                run_tutor(client, system_prompt, llm_history),
                TUTOR_TIMEOUT_S,
                label=f"tutoring (case={case_id}, turn={t_i})",
            )

            assistant_outputs.append(assistant_text)
            llm_history.append({"role": "assistant", "content": assistant_text})

        transcript = format_transcript(turns, assistant_outputs)

        if verbose:
            print("[EVAL]   Judging...")

        judge_user = build_judge_user_prompt(rubric, transcript)
        judge_resp = await _with_timeout(
            client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": judge_user},
                ],
                temperature=JUDGE_TEMP,
                response_format={"type": "json_object"},
            ),
            JUDGE_TIMEOUT_S,
            label=f"judging (case={case_id})",
        )

        judge_content = judge_resp.choices[0].message.content or "{}"
        try:
            judge_json = json.loads(judge_content)
        except json.JSONDecodeError:
            judge_json = {"error": "judge_json_parse_failed", "raw": judge_content}

        results.append(
            {
                "case_id": case_id,
                "phase_target": case.get("phase_target"),
                "notes": case.get("notes"),
                "final_phase": current_phase,
                "transcript": transcript,
                "judge": judge_json,
            }
        )

        if verbose:
            overall = judge_json.get("overall_score", None)
            print(f"[EVAL]   Done. overall_score={overall}")

    base_name = f"report_{Path(suite_file).stem}__{Path(rubric_file).stem}"
    version = 0

    while True:
        report_path = REPORTS_DIR / f"{base_name}_v{version}.json"
        if not report_path.exists():
            break
        version += 1

    report_data = {
        "metadata": {
            "suite_file": suite_file,
            "rubric_file": rubric_file,
            "tutor_type": tutor_type,
            "base_srl_version": BASE_SRL_VERSION,
            "forethought_version": SRL_PHASE_VERSION,
            "judge_model": JUDGE_MODEL,
        },
        "results": results,
    }

    report_path.write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if verbose:
        print(f"\n[EVAL] Wrote report: {report_path}")

    return str(report_path)
