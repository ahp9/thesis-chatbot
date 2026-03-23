import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from evaluator.judge_v1 import JUDGE_SYSTEM, build_judge_user_prompt
from services.llm_client import get_client
from services.router import route_message
from services.srl_chain import run_srl_chain

ROOT = Path(__file__).resolve().parents[1]
RUBRICS_DIR = ROOT / "evaluator" / "rubrics"
PROMPTS_DIR = ROOT / "evaluator" / "prompts"
REPORTS_DIR = ROOT / "evaluator" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

JUDGE_MODEL = "gpt-4o-mini"
JUDGE_TEMP = 0.0

# --- Version tracking ---
# Update these when you change the corresponding prompt file so reports are
# automatically stamped with the exact configuration that produced them.
VERSION_BASE_SRL    = "v5"   # srl_model_v{N}.txt 
VERSION_ROUTER      = "v5"   # router_system_prompt_v{N}.txt 
VERSION_CHAIN       = "v2"   # srl_chain.py / chain prompt files 
VERSION_RESPOND     = "v2"   # response_generation_prompt_v{N}.txt 
VERSION_FORETHOUGHT = "v4"   # forethought_core.txt 
VERSION_PERFORMANCE = "v6"   # performance_core.txt 
VERSION_REFLECTION  = "v2"   # reflection_core.txt

PHASE_ORDER = ["FORETHOUGHT", "PERFORMANCE", "REFLECTION"]

# Timeouts (seconds)
ROUTE_TIMEOUT_S = 45
CHAIN_TIMEOUT_S = 120
JUDGE_TIMEOUT_S = 90


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_suite(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError(f"Prompt suite file is empty: {path}")
    return json.loads(raw)


def require_file(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{kind} file not found: {path}")


def format_transcript(
    turns: List[Dict[str, str]], assistant_turns: List[str]
) -> str:
    out: List[str] = []
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
        raise TimeoutError(f"{label} timed out after {seconds} seconds") from e


def _update_phase(
    current_phase: str, predicted_phase: str, confidence: float
) -> str:
    current_phase = (current_phase or "FORETHOUGHT").upper()
    predicted_phase = (predicted_phase or current_phase).upper()
    confidence = float(confidence or 0.0)

    if current_phase not in PHASE_ORDER:
        current_phase = "FORETHOUGHT"
    if predicted_phase not in PHASE_ORDER:
        predicted_phase = current_phase

    if confidence < 0.60:
        return current_phase

    if PHASE_ORDER.index(predicted_phase) >= PHASE_ORDER.index(current_phase):
        return predicted_phase

    return current_phase


def _build_version_tag() -> str:
    return (
        f"base-{VERSION_BASE_SRL}"
        f"_router-{VERSION_ROUTER}"
        f"_chain-{VERSION_CHAIN}"
        f"_ft-{VERSION_FORETHOUGHT}"
        f"_perf-{VERSION_PERFORMANCE}"
        f"_refl-{VERSION_REFLECTION}"
    )


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

async def run_suite(
    suite_file: str,
    rubric_file: str,
    tutor_type: str = "SRL Tutor",
    max_cases: Optional[int] = None,
    verbose: bool = True,
) -> str:
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
        print(f"[EVAL] Suite  : {suite_path} ({len(suite)} cases)")
        print(f"[EVAL] Rubric : {rubric_path}  type={rubric.get('rubric_type', 'response')}")
        print(f"[EVAL] Tutor  : {tutor_type}")
        print(f"[EVAL] Versions: {_build_version_tag()}")
        print(
            f"[EVAL] Timeouts: route={ROUTE_TIMEOUT_S}s "
            f"chain={CHAIN_TIMEOUT_S}s judge={JUDGE_TIMEOUT_S}s"
        )

    results: List[Dict[str, Any]] = []

    for idx, case in enumerate(suite, start=1):
        case_id = case.get("id", f"case_{idx}")
        turns: List[Dict[str, str]] = case["turns"]
        llm_history: List[Dict[str, str]] = []
        assistant_outputs: List[str] = []
        turn_chain_snapshots: List[Dict[str, Any]] = []

        current_phase = case.get("start_phase", "FORETHOUGHT").upper()

        if verbose:
            print(
                f"\n[EVAL] Case {idx}/{len(suite)}: {case_id} "
                f"(start_phase={current_phase}, turns={len(turns)})"
            )

        for t_i, turn in enumerate(turns, start=1):
            if turn["role"] != "user":
                continue

            user_msg = turn["content"]
            llm_history.append({"role": "user", "content": user_msg})

            route: Dict[str, Any] = {
                "phase": current_phase,
                "strategy": "NONE",
                "confidence": 0.0,
                "signals": [],
            }

            if tutor_type == "SRL Tutor":
                if verbose:
                    print(f"[EVAL]   Turn {t_i}: routing...")

                route = await _with_timeout(
                    route_message(client, user_msg, llm_history, current_phase),
                    seconds=ROUTE_TIMEOUT_S,
                    label=f"routing (case={case_id}, turn={t_i})",
                )

                predicted_phase = route.get("phase", current_phase)
                conf = float(route.get("confidence", 0.0))
                current_phase = _update_phase(current_phase, predicted_phase, conf)
                route["phase"] = current_phase

                if verbose:
                    print(
                        f"[EVAL]   Turn {t_i}: "
                        f"route.phase={route['phase']} "
                        f"conf={conf:.2f} "
                        f"strategy={route.get('strategy', 'NONE')}"
                    )

            if verbose:
                print(f"[EVAL]   Turn {t_i}: running chain...")

            chain_result = await _with_timeout(
                run_srl_chain(client, route, llm_history, user_msg),
                CHAIN_TIMEOUT_S,
                label=f"chain (case={case_id}, turn={t_i})",
            )

            final_reply: str = chain_result["reply"]
            draft_reply: str = chain_result.get("draft_reply", final_reply)
            was_rewritten: bool = bool(chain_result.get("was_rewritten", final_reply != draft_reply))

            chain_internals: Dict[str, Any] = {
                "route": route,
                "diagnosis": chain_result.get("diagnosis", {}),
                "decision": chain_result.get("decision", {}),
                "check": chain_result.get("check", {}),
                "draft_reply": draft_reply,
                "final_reply": final_reply,
                "was_rewritten": was_rewritten,
            }
            turn_chain_snapshots.append(
                {"turn": t_i, "user_message": user_msg, **chain_internals}
            )

            if verbose:
                support = chain_internals["decision"].get("support_level", "?")
                leaks = chain_internals["check"].get("leaks_solution", "?")
                print(
                    f"[EVAL]   Turn {t_i}: support={support} "
                    f"rewritten={was_rewritten} "
                    f"leaks={leaks}"
                )

            assistant_outputs.append(final_reply)
            llm_history.append({"role": "assistant", "content": final_reply})

        transcript = format_transcript(turns, assistant_outputs)

        aggregate_chain_internals: Dict[str, Any] = {
            "turn_snapshots": turn_chain_snapshots,
            **(turn_chain_snapshots[-1] if turn_chain_snapshots else {}),
        }

        if verbose:
            print("[EVAL]   Judging...")

        judge_user = build_judge_user_prompt(
            rubric, transcript, chain_internals=aggregate_chain_internals
        )

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
            judge_json = {
                "error": "judge_json_parse_failed",
                "raw": judge_content,
            }

        results.append(
            {
                "case_id": case_id,
                "phase_target": case.get("phase_target"),
                "expected_support_level": case.get("expected_support_level"),
                "expected_request_kind": case.get("expected_request_kind"),
                "notes": case.get("notes"),
                "final_phase": current_phase,
                "transcript": transcript,
                "chain_snapshots": turn_chain_snapshots,
                "judge": judge_json,
            }
        )

        if verbose:
            overall = judge_json.get("overall_score")
            flags = judge_json.get("fail_flags", [])
            print(f"[EVAL]   Done. overall_score={overall}  fail_flags={flags}")

    version_tag = _build_version_tag()
    base_name = (
        f"report"
        f"__{Path(suite_file).stem}"
        f"__{Path(rubric_file).stem}"
        f"__{version_tag}"
    )
    run_index = 0
    while True:
        report_path = REPORTS_DIR / f"{base_name}__v{run_index}.json"
        if not report_path.exists():
            break
        run_index += 1

    report_data: Dict[str, Any] = {
        "metadata": {
            "suite_file": suite_file,
            "rubric_file": rubric_file,
            "rubric_type": rubric.get("rubric_type", "response"),
            "tutor_type": tutor_type,
            "judge_model": JUDGE_MODEL,
            "versions": {
                "base_srl": VERSION_BASE_SRL,
                "router": VERSION_ROUTER,
                "chain": VERSION_CHAIN,
                "respond": VERSION_RESPOND,
                "forethought": VERSION_FORETHOUGHT,
                "performance": VERSION_PERFORMANCE,
                "reflection": VERSION_REFLECTION,
            },
        },
        "results": results,
    }

    report_path.write_text(
        json.dumps(report_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if verbose:
        print(f"\n[EVAL] Wrote report: {report_path}")

    return str(report_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run an SRL tutor evaluation suite and write a JSON report."
    )
    p.add_argument(
        "--suite",
        required=True,
        metavar="FILE",
        help="Prompt suite JSON filename inside evaluator/prompts/  e.g. forethought_suite.json",
    )
    p.add_argument(
        "--rubric",
        required=True,
        metavar="FILE",
        help="Rubric YAML filename inside evaluator/rubrics/  e.g. rubric_response.yaml",
    )
    p.add_argument(
        "--tutor",
        default="SRL Tutor",
        metavar="TYPE",
        help='Tutor type string passed to the chain. Default: "SRL Tutor"',
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of cases evaluated (useful for quick smoke tests).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-turn progress output.",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    asyncio.run(
        run_suite(
            suite_file=args.suite,
            rubric_file=args.rubric,
            tutor_type=args.tutor,
            max_cases=args.max_cases,
            verbose=not args.quiet,
        )
    )