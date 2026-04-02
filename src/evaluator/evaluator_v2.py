import argparse
import asyncio
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from evaluator.judge_v1 import JUDGE_SYSTEM, build_judge_user_prompt
from lib.enums import Phase
from services.llm_client import get_client
from services.orchestrator import Orchestrator

ROOT = Path(__file__).resolve().parents[1]
RUBRICS_DIR   = ROOT / "evaluator" / "rubrics"
PERSONAS_DIR  = ROOT / "evaluator" / "prompts"
REPORTS_DIR   = ROOT / "evaluator" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Default personas file 
DEFAULT_PERSONAS_FILE = "personas.json"

# Version constants
VERSION_BASE_SRL    = "v7"
VERSION_ROUTER      = "v7"
VERSION_CHAIN       = "v5"
VERSION_RESPOND     = "v2"
VERSION_FORETHOUGHT = "v6"
VERSION_PERFORMANCE = "v6"
VERSION_REFLECTION  = "v2"

# Model config
STUDENT_MODEL  = "gpt-4o-mini"   # Simulates the student
JUDGE_MODEL    = "gpt-4o-mini"   # Same as evaluator_v1.py
STUDENT_TEMP   = 0.8             # Higher temp → more naturalistic student variance
JUDGE_TEMP     = 0.0

# Timeouts (seconds)
ORCHESTRATOR_TIMEOUT_S = 180
STUDENT_TIMEOUT_S      = 60
JUDGE_TIMEOUT_S        = 90


# ---------------------------------------------------------------------------
# Student simulator prompt
# ---------------------------------------------------------------------------

STUDENT_SYSTEM_TEMPLATE = """\
You are roleplaying as a student in a one-on-one tutoring session.

PERSONA
-------
Name (optional):  {name}
Subject / Task:   {topic}
Your situation:   {state_description}
Expertise level:  {expertise_level}   (NOVICE | INTERMEDIATE | ADVANCED)
Request kind:     {request_kind}      (PRODUCT = you want to produce something | RESOURCE = you want to understand something)
Emotional state:  {emotional_state}

BEHAVIOUR RULES
---------------
1. Respond ONLY as the student.  Never break character.
2. Keep messages SHORT — 1 to 4 sentences, exactly like a real chat message.
3. Reflect the tutor's reply naturally.
   - If the tutor gave a useful hint, show partial progress: attempt something,
     even if imperfect.
   - If the tutor asked you a question, answer it (possibly incorrectly if
     your expertise is NOVICE).
   - If the tutor's reply was confusing or too abstract, express confusion.
   - If the tutor repeated the same unhelpful support, show mild frustration.
4. Do NOT improve too fast.  A NOVICE should still struggle after one hint.
5. Do NOT ask for the solution outright unless you have been stuck for 3+
   turns with no progress.
6. Show realistic misconceptions appropriate to your expertise level.
7. If the tutor gave you complete code or a full answer, express relief and
   say you will submit/use it — this is evidence of a pedagogical failure and
   the judge needs to see it.
8. Stay within the task domain — do not introduce new unrelated topics.

OPENING MESSAGE GUIDANCE
------------------------
Your very first message should naturally introduce your situation.
It should match your state_description above and request_kind.
Do not just paste your state_description — phrase it as a real student would.
"""

STUDENT_CONTINUE_TEMPLATE = """\
The tutor just replied:

---
{tutor_reply}
---

Respond as the student.  Remember your situation: {state_description}
Your expertise is {expertise_level}.  Keep it to 1–4 sentences.
"""


# Helpers 
def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_personas(path: Path) -> List[Dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError(f"Personas file is empty: {path}")
    return json.loads(raw)


def require_file(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")


def _build_version_tag() -> str:
    return (
        f"base-{VERSION_BASE_SRL}"
        f"_router-{VERSION_ROUTER}"
        f"_chain-{VERSION_CHAIN}"
        f"_ft-{VERSION_FORETHOUGHT}"
        f"_perf-{VERSION_PERFORMANCE}"
        f"_refl-{VERSION_REFLECTION}"
    )


async def _with_timeout(coro, seconds: int, label: str):
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except asyncio.TimeoutError as e:
        raise TimeoutError(f"{label} timed out after {seconds}s") from e


def _format_dynamic_transcript(
    simulated_turns: List[Dict[str, str]]
) -> str:
    """
    Build a transcript string from the recorded turn pairs.
    """
    lines = []
    for entry in simulated_turns:
        lines.append(f"USER: {entry['student']}")
        if "tutor" in entry:
            lines.append(f"ASSISTANT: {entry['tutor']}")
    return "\n".join(lines)


# Student simulator
async def _generate_student_turn(
    client,
    persona: Dict[str, Any],
    tutor_reply: Optional[str],
    turn_index: int,
) -> str:
    """
    Call the student-simulator LLM to produce the next student message.

    turn_index == 1  → opening message (no prior tutor reply)
    turn_index  > 1  → reactive reply to tutor_reply
    """
    system_prompt = STUDENT_SYSTEM_TEMPLATE.format(
        name=persona.get("name", ""),
        topic=persona["topic"],
        state_description=persona["state_description"],
        expertise_level=persona.get("expertise_level", "INTERMEDIATE"),
        request_kind=persona.get("request_kind", "PRODUCT"),
        emotional_state=persona.get("emotional_state", "Neutral — slightly uncertain"),
    )

    if turn_index == 1:
        user_content = (
            "Start the tutoring session.  Write your opening message to the tutor."
        )
    else:
        user_content = STUDENT_CONTINUE_TEMPLATE.format(
            tutor_reply=tutor_reply,
            state_description=persona["state_description"],
            expertise_level=persona.get("expertise_level", "INTERMEDIATE"),
        )

    resp = await _with_timeout(
        client.chat.completions.create(
            model=STUDENT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            temperature=STUDENT_TEMP,
            max_tokens=300,
        ),
        STUDENT_TIMEOUT_S,
        label=f"student simulator (turn={turn_index})",
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Core dynamic evaluation loop
# ---------------------------------------------------------------------------

async def _run_persona(
    client,
    orchestrator: Orchestrator,
    rubric: Dict[str, Any],
    persona: Dict[str, Any],
    persona_idx: int,
    total: int,
    max_turns: int,
    verbose: bool,
) -> Dict[str, Any]:
    """
    Run one persona through the full student-simulator → tutor → judge pipeline.
    Returns a result dict compatible with evaluator_v1.py's results list.
    """
    persona_id = persona.get("id", f"persona_{persona_idx}")
    start_phase_str = (persona.get("start_phase") or "FORETHOUGHT").upper()
    current_phase   = Phase(start_phase_str)

    if verbose:
        print(
            f"\n[DYN] Persona {persona_idx}/{total}: {persona_id}"
            f"  topic={persona['topic'][:50]}"
            f"  start_phase={current_phase.value}"
            f"  turns={max_turns}"
        )

    llm_history:        List[Dict[str, str]] = []
    simulated_turns:    List[Dict[str, str]] = []   # [{student, tutor}, ...]
    turn_chain_snapshots: List[Dict[str, Any]] = []
    chain_internals:    Dict[str, Any] = {}

    tutor_reply: Optional[str] = None

    for turn_i in range(1, max_turns + 1):

        # 1. Student generates their message 
        if verbose:
            print(f"[DYN]   Turn {turn_i}/{max_turns}: generating student message...")

        student_msg = await _generate_student_turn(
            client, persona, tutor_reply, turn_index=turn_i
        )

        if verbose:
            print(f"[DYN]   Student: {student_msg[:120]}{'...' if len(student_msg)>120 else ''}")

        simulated_turns.append({"student": student_msg})
        llm_history.append({"role": "user", "content": student_msg})

        # 2. SRL tutor responds
        if verbose:
            print(f"[DYN]   Turn {turn_i}/{max_turns}: running orchestrator...")

        result = await _with_timeout(
            orchestrator.handle_turn(
                user_message=student_msg,
                llm_history=llm_history,
                current_phase=current_phase,
            ),
            ORCHESTRATOR_TIMEOUT_S,
            label=f"orchestrator (persona={persona_id}, turn={turn_i})",
        )

        current_phase  = result.route.phase
        tutor_reply    = result.reply
        draft_reply    = result.draft_reply or tutor_reply
        was_rewritten  = bool(result.was_rewritten)

        chain_internals = {
            "route":        result.route.to_dict(),
            "diagnosis":    result.control.checkpoint.to_dict(),
            "decision":     result.control.decision.to_dict(),
            "check":        result.safety.to_dict(),
            "draft_reply":  draft_reply,
            "final_reply":  tutor_reply,
            "was_rewritten": was_rewritten,
        }

        turn_chain_snapshots.append({
            "turn":         turn_i,
            "user_message": student_msg,
            **chain_internals,
        })

        simulated_turns[-1]["tutor"] = tutor_reply

        if verbose:
            support = chain_internals["decision"].get("support_level", "?")
            leaks   = chain_internals["check"].get("leaks_solution", "?")
            print(
                f"[DYN]   Tutor:   support={support}  "
                f"rewritten={was_rewritten}  leaks={leaks}"
            )
            print(f"[DYN]   Reply:   {tutor_reply[:120]}{'...' if len(tutor_reply)>120 else ''}")

        llm_history.append({
            "role":      "assistant",
            "content":   tutor_reply,
            "route":     chain_internals["route"],
            "diagnosis": chain_internals["diagnosis"],
            "decision":  chain_internals["decision"],
            "check":     chain_internals["check"],
            "draft_reply": draft_reply,
        })

        # ── 3. Optional early stop: task is done ─────────────────────────────
        # If the student explicitly says they have everything they need and
        # the phase is REFLECTION, no point running more turns.
        if (
            current_phase == Phase.REFLECTION
            and turn_i >= 3
        ):
            if verbose:
                print(f"[DYN]   Phase reached REFLECTION at turn {turn_i} — ending early.")
            break

    # 4. Judge the full conversation 
    transcript = _format_dynamic_transcript(simulated_turns)

    # AFTER — only structured multi-turn data, no last-turn bias
    aggregate_chain_internals: Dict[str, Any] = {
        "turn_snapshots": turn_chain_snapshots,
        "total_turns":    len(turn_chain_snapshots),
        "final_phase":    current_phase.value,
    }

    if verbose:
        print(f"[DYN]   Judging ({len(simulated_turns)}-turn conversation)...")

    judge_user = build_judge_user_prompt(
        rubric, transcript, chain_internals=aggregate_chain_internals
    )

    judge_resp = await _with_timeout(
        client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": judge_user},
            ],
            temperature=JUDGE_TEMP,
            response_format={"type": "json_object"},
        ),
        JUDGE_TIMEOUT_S,
        label=f"judge (persona={persona_id})",
    )

    judge_content = judge_resp.choices[0].message.content or "{}"
    try:
        judge_json = json.loads(judge_content)
    except json.JSONDecodeError:
        judge_json = {"error": "judge_json_parse_failed", "raw": judge_content}

    if verbose:
        overall = judge_json.get("overall_score")
        flags   = judge_json.get("fail_flags", [])
        print(f"[DYN]   Done.  overall_score={overall}  fail_flags={flags}")

    return {
        "case_id":                 persona_id,
        "phase_target":            persona.get("phase_target"),
        "expected_support_level":  persona.get("expected_support_level"),
        "expected_request_kind":   persona.get("expected_request_kind"),
        "notes":                   persona.get("notes"),
        "final_phase":             current_phase.value,
        "transcript":              transcript,
        "chain_snapshots":         turn_chain_snapshots,
        "judge":                   judge_json,
        # Dynamic-eval extras
        "persona":                 persona,
        "simulated_turns":         simulated_turns,
        "turns_completed":         len(simulated_turns),
    }



# Comparison helper
def _compare_reports(
    dynamic_results:  List[Dict[str, Any]],
    static_report_path: str,
    verbose: bool,
) -> Dict[str, Any]:
    """
    Load a static evaluator_v1.py report and produce a side-by-side summary.
    Returns a dict with aggregate score stats for both modes.
    """
    static_path = Path(static_report_path)
    if not static_path.exists():
        if verbose:
            print(f"[CMP] Static report not found: {static_path}  — skipping comparison.")
        return {}

    static_data    = json.loads(static_path.read_text(encoding="utf-8"))
    static_results = static_data.get("results", [])

    def _scores(results: List[Dict[str, Any]]) -> List[float]:
        out = []
        for r in results:
            s = r.get("judge", {}).get("overall_score")
            if s is not None:
                try:
                    out.append(float(s))
                except (TypeError, ValueError):
                    pass
        return out

    def _flag_rate(results: List[Dict[str, Any]]) -> float:
        flagged = sum(
            1 for r in results
            if r.get("judge", {}).get("fail_flags")
        )
        return round(flagged / len(results), 3) if results else 0.0

    dyn_scores    = _scores(dynamic_results)
    static_scores = _scores(static_results)

    def _stats(scores: List[float]) -> Dict[str, Any]:
        if not scores:
            return {"n": 0, "mean": None, "median": None, "stdev": None}
        return {
            "n":      len(scores),
            "mean":   round(statistics.mean(scores),   2),
            "median": round(statistics.median(scores), 2),
            "stdev":  round(statistics.stdev(scores),  2) if len(scores) > 1 else 0.0,
        }

    comparison = {
        "dynamic": {
            **_stats(dyn_scores),
            "flag_rate": _flag_rate(dynamic_results),
        },
        "static": {
            **_stats(static_scores),
            "flag_rate":   _flag_rate(static_results),
            "source_file": str(static_path.name),
        },
        "delta_mean": (
            round(_stats(dyn_scores)["mean"] - _stats(static_scores)["mean"], 2)
            if dyn_scores and static_scores
            else None
        ),
        "interpretation": (
            "Dynamic scores are LOWER — the simulator found harder cases the static suite missed."
            if (dyn_scores and static_scores and statistics.mean(dyn_scores) < statistics.mean(static_scores))
            else (
                "Dynamic scores are SIMILAR or HIGHER — the simulator did not reveal systematic gaps vs. the static suite."
                if dyn_scores and static_scores
                else "Insufficient data for comparison."
            )
        ),
    }

    if verbose:
        print("\n[CMP] ── Score Comparison ──────────────────────────────────")
        print(f"[CMP]  Dynamic  : mean={comparison['dynamic']['mean']}  "
              f"median={comparison['dynamic']['median']}  "
              f"stdev={comparison['dynamic']['stdev']}  "
              f"flag_rate={comparison['dynamic']['flag_rate']}")
        print(f"[CMP]  Static   : mean={comparison['static']['mean']}  "
              f"median={comparison['static']['median']}  "
              f"stdev={comparison['static']['stdev']}  "
              f"flag_rate={comparison['static']['flag_rate']}")
        print(f"[CMP]  Δ mean   : {comparison['delta_mean']}")
        print(f"[CMP]  {comparison['interpretation']}")
        print("[CMP] ────────────────────────────────────────────────────────")

    return comparison


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run_dynamic(
    rubric_file: str,
    personas_file: str = DEFAULT_PERSONAS_FILE,
    tutor_type: str = "SRL Tutor",
    max_cases: Optional[int] = None,
    max_turns_per_case: int = 6,
    verbose: bool = True,
    compare_report: Optional[str] = None,
) -> str:
    client = get_client()

    rubric_path   = RUBRICS_DIR / rubric_file
    personas_path = PERSONAS_DIR / personas_file
    require_file(rubric_path, "Rubric")
    require_file(personas_path, "Personas file")

    rubric   = load_yaml(rubric_path)
    personas = load_personas(personas_path)

    if max_cases is not None:
        personas = personas[:max_cases]

    if verbose:
        print(f"[DYN] Personas : {personas_path} ({len(personas)} cases)")
        print(f"[DYN] Rubric   : {rubric_path}  type={rubric.get('rubric_type', 'response')}")
        print(f"[DYN] Tutor    : {tutor_type}")
        print(f"[DYN] Max turns: {max_turns_per_case}")
        print(f"[DYN] Versions : {_build_version_tag()}")

    orchestrator = Orchestrator(client)
    results: List[Dict[str, Any]] = []
    failed_cases: List[Dict[str, Any]] = []

    version_tag = _build_version_tag()
    base_name = (
        f"report"
        f"__dynamic"
        f"__{Path(personas_file).stem}"
        f"__{Path(rubric_file).stem}"
        f"__{version_tag}"
    )
    run_index = 0
    while True:
        report_path = REPORTS_DIR / f"{base_name}__v{run_index}.json"
        if not report_path.exists():
            break
        run_index += 1

    def _write_checkpoint_report():
        report_data: Dict[str, Any] = {
            "metadata": {
                "eval_mode": "dynamic",
                "personas_file": personas_file,
                "rubric_file": rubric_file,
                "rubric_type": rubric.get("rubric_type", "response"),
                "tutor_type": tutor_type,
                "student_model": STUDENT_MODEL,
                "judge_model": JUDGE_MODEL,
                "max_turns_per_case": max_turns_per_case,
                "versions": {
                    "base_srl": VERSION_BASE_SRL,
                    "router": VERSION_ROUTER,
                    "chain": VERSION_CHAIN,
                    "respond": VERSION_RESPOND,
                    "forethought": VERSION_FORETHOUGHT,
                    "performance": VERSION_PERFORMANCE,
                    "reflection": VERSION_REFLECTION,
                },
                "completed_cases": len(results),
                "failed_cases": len(failed_cases),
                "total_cases_requested": len(personas),
            },
            "results": results,
            "failures": failed_cases,
        }

        report_path.write_text(
            json.dumps(report_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    for idx, persona in enumerate(personas, start=1):
        try:
            result = await _run_persona(
                client=client,
                orchestrator=orchestrator,
                rubric=rubric,
                persona=persona,
                persona_idx=idx,
                total=len(personas),
                max_turns=max_turns_per_case,
                verbose=verbose,
            )
            results.append(result)

        except Exception as e:
            failed_cases.append({
                "case_id": persona.get("id", f"persona_{idx}"),
                "persona_index": idx,
                "error_type": type(e).__name__,
                "error": str(e),
                "persona": persona,
            })
            if verbose:
                print(f"[DYN] ERROR in case {idx}/{len(personas)}: {persona.get('id')} -> {type(e).__name__}: {e}")

        finally:
            _write_checkpoint_report()

    if verbose:
        print(f"\n[DYN] Wrote report: {report_path}")

    return str(report_path)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run the SRL tutor against a dynamic student simulator "
            "and evaluate the resulting conversations."
        )
    )
    p.add_argument(
        "--rubric",
        required=True,
        metavar="FILE",
        help="Rubric YAML filename inside evaluator/rubrics/  e.g. rubric_response.yaml",
    )
    p.add_argument(
        "--personas",
        default=DEFAULT_PERSONAS_FILE,
        metavar="FILE",
        help=f"Personas JSON filename inside evaluator/prompts/  (default: {DEFAULT_PERSONAS_FILE})",
    )
    p.add_argument(
        "--tutor",
        default="SRL Tutor",
        metavar="TYPE",
        help='Tutor type string passed to the chain.  Default: "SRL Tutor"',
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of personas evaluated.",
    )
    p.add_argument(
        "--max-turns",
        type=int,
        default=3,
        metavar="N",
        help="Maximum turns per conversation (default: 6).",
    )
    p.add_argument(
        "--compare-report",
        default=None,
        metavar="PATH",
        help=(
            "Path to an existing evaluator_v1.py JSON report.  "
            "When provided, prints a side-by-side score comparison."
        ),
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
        run_dynamic(
            rubric_file=args.rubric,
            personas_file=args.personas,
            tutor_type=args.tutor,
            max_cases=args.max_cases,
            max_turns_per_case=args.max_turns,
            verbose=not args.quiet,
            compare_report=args.compare_report,
        )
    )