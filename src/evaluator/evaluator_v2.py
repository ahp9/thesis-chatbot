import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from evaluator.judge_v1 import JUDGE_SYSTEM, build_judge_user_prompt
from lib.enums import Phase
from services.llm_client import get_client
from services.orchestrator import Orchestrator

ROOT = Path(__file__).resolve().parents[1]
RUBRICS_DIR  = ROOT / "evaluator" / "rubrics"
PERSONAS_DIR = ROOT / "evaluator" / "prompts"
REPORTS_DIR  = ROOT / "evaluator" / "reports"
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
STUDENT_MODEL = "gpt-4o-mini"    # Simulates the student — short outputs, cheap
JUDGE_MODEL   = "gpt-4.1-mini"   
STUDENT_TEMP  = 0.8              # Higher temp → more naturalistic student variance
JUDGE_TEMP    = 0.0

# How many persona cases run simultaneously.
CONCURRENCY = 2

# Timeouts (seconds)
ORCHESTRATOR_TIMEOUT_S = 180
STUDENT_TIMEOUT_S      = 60
JUDGE_TIMEOUT_S        = 120   

# Judge retry config 
JUDGE_MAX_RETRIES = 2   # Total attempts (1 original + 1 retry)


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


def _format_dynamic_transcript(simulated_turns: List[Dict[str, str]]) -> str:
    """Build a transcript string from the recorded turn pairs."""
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
        user_content = "Start the tutoring session.  Write your opening message to the tutor."
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


# Judge call with retry
async def _call_judge(
    client,
    judge_user: str,
    persona_id: str,
    verbose: bool,
) -> Dict[str, Any]:
    """
    Call the judge model and parse JSON.
    Retries once on JSON parse failure before giving up.
    Returns the parsed judge dict, or an error dict if all attempts fail.
    """
    last_raw = ""
    for attempt in range(JUDGE_MAX_RETRIES):
        try:
            resp = await _with_timeout(
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
                label=f"judge attempt {attempt + 1} (persona={persona_id})",
            )
            content = resp.choices[0].message.content or "{}"
            last_raw = content
            return json.loads(content)

        except json.JSONDecodeError:
            if verbose:
                print(
                    f"[DYN]   Judge JSON parse failed "
                    f"(attempt {attempt + 1}/{JUDGE_MAX_RETRIES}) — "
                    f"{'retrying...' if attempt + 1 < JUDGE_MAX_RETRIES else 'giving up.'}"
                )
        except Exception as e:
            # Timeout or API error — don't retry, propagate
            raise e

    # All retries exhausted
    return {
        "error":      "judge_json_parse_failed",
        "attempts":   JUDGE_MAX_RETRIES,
        "raw":        last_raw[:500],
    }


# Single persona run
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
    Returns a result dict.
    """
    persona_id      = persona.get("id", f"persona_{persona_idx}")
    start_phase_str = (persona.get("start_phase") or "FORETHOUGHT").upper()
    current_phase   = Phase(start_phase_str)

    if verbose:
        print(
            f"\n[DYN] Persona {persona_idx}/{total}: {persona_id}"
            f"  topic={persona['topic'][:50]}"
            f"  start_phase={current_phase.value}"
            f"  turns={max_turns}"
        )

    llm_history:          List[Dict[str, str]] = []
    simulated_turns:      List[Dict[str, str]] = []
    turn_chain_snapshots: List[Dict[str, Any]] = []
    tutor_reply:          Optional[str]        = None

    for turn_i in range(1, max_turns + 1):

        # 1. Student generates their message
        if verbose:
            print(f"[DYN]   Turn {turn_i}/{max_turns}: generating student message...")

        student_msg = await _generate_student_turn(
            client, persona, tutor_reply, turn_index=turn_i
        )

        if verbose:
            print(f"[DYN]   Student: {student_msg[:120]}{'...' if len(student_msg) > 120 else ''}")

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

        current_phase = result.route.phase
        tutor_reply   = result.reply
        draft_reply   = result.draft_reply or tutor_reply
        was_rewritten = bool(result.was_rewritten)

        chain_internals = {
            "route":         result.route.to_dict(),
            "diagnosis":     result.control.checkpoint.to_dict(),
            "decision":      result.control.decision.to_dict(),
            "check":         result.safety.to_dict(),
            "draft_reply":   draft_reply,
            "final_reply":   tutor_reply,
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
            print(f"[DYN]   Reply:   {tutor_reply[:120]}{'...' if len(tutor_reply) > 120 else ''}")

        llm_history.append({
            "role":        "assistant",
            "content":     tutor_reply,
            "route":       chain_internals["route"],
            "diagnosis":   chain_internals["diagnosis"],
            "decision":    chain_internals["decision"],
            "check":       chain_internals["check"],
            "draft_reply": draft_reply,
        })

        # 3. Optional early stop: task is done
        if current_phase == Phase.REFLECTION and turn_i >= 3:
            if verbose:
                print(f"[DYN]   Phase reached REFLECTION at turn {turn_i} — ending early.")
            break

    # 4. Judge the full conversation
    transcript = _format_dynamic_transcript(simulated_turns)

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

    judge_json = await _call_judge(client, judge_user, persona_id, verbose)

    if verbose:
        overall = judge_json.get("overall_score")
        flags   = judge_json.get("fail_flags", [])
        err     = judge_json.get("error")
        if err:
            print(f"[DYN]   Done.  JUDGE ERROR: {err}")
        else:
            print(f"[DYN]   Done.  overall_score={overall}  fail_flags={flags}")

    return {
        "case_id":                persona_id,
        "phase_target":           persona.get("phase_target"),
        "expected_support_level": persona.get("expected_support_level"),
        "expected_request_kind":  persona.get("expected_request_kind"),
        "notes":                  persona.get("notes"),
        "final_phase":            current_phase.value,
        "transcript":             transcript,
        "chain_snapshots":        turn_chain_snapshots,
        "judge":                  judge_json,
        "persona":                persona,
        "simulated_turns":        simulated_turns,
        "turns_completed":        len(simulated_turns),
    }

# AI judge full run
async def run_dynamic(
    rubric_file:         str,
    personas_file:       str          = DEFAULT_PERSONAS_FILE,
    tutor_type:          str          = "SRL Tutor",
    max_cases:           Optional[int] = None,
    max_turns_per_case:  int          = 6,
    concurrency:         int          = CONCURRENCY,
    verbose:             bool         = True,
    resume_from:         Optional[str] = None,
) -> str:
    """
    Run the full dynamic evaluation pipeline.

    Parameters
    ----------
    rubric_file         : YAML filename inside evaluator/rubrics/
    personas_file       : JSON filename inside evaluator/prompts/
    tutor_type          : String passed to the orchestrator
    max_cases           : Cap personas evaluated (None = all)
    max_turns_per_case  : Max turns per simulated conversation
    concurrency         : How many cases run simultaneously (default: CONCURRENCY)
    verbose             : Print per-turn progress
    compare_report      : Path to a static report for side-by-side comparison
    resume_from         : Path to a prior checkpoint report — skip already-completed cases
    """
    client = get_client()

    rubric_path   = RUBRICS_DIR / rubric_file
    personas_path = PERSONAS_DIR / personas_file
    require_file(rubric_path,   "Rubric")
    require_file(personas_path, "Personas file")

    rubric   = load_yaml(rubric_path)
    personas = load_personas(personas_path)

    if max_cases is not None:
        personas = personas[:max_cases]

    # Resume: skip cases already in a prior checkpoint
    completed_ids: set = set()
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            prior = json.loads(resume_path.read_text(encoding="utf-8"))
            completed_ids = {r["case_id"] for r in prior.get("results", [])}
            if verbose and completed_ids:
                print(f"[DYN] Resuming — skipping {len(completed_ids)} already-completed cases.")
        else:
            if verbose:
                print(f"[DYN] Resume path not found ({resume_from}) — starting fresh.")

    personas_to_run = [p for p in personas if p.get("id") not in completed_ids]

    if verbose:
        print(f"[DYN] Personas : {personas_path} ({len(personas)} total, {len(personas_to_run)} to run)")
        print(f"[DYN] Rubric   : {rubric_path}  type={rubric.get('rubric_type', 'response')}")
        print(f"[DYN] Tutor    : {tutor_type}")
        print(f"[DYN] Max turns: {max_turns_per_case}")
        print(f"[DYN] Concurrency: {concurrency}")
        print(f"[DYN] Judge model: {JUDGE_MODEL}")
        print(f"[DYN] Versions : {_build_version_tag()}")

    orchestrator = Orchestrator(client)

    # Pre-populate results with any resumed cases
    results:      List[Dict[str, Any]] = []
    failed_cases: List[Dict[str, Any]] = []

    if resume_from and completed_ids:
        resume_path = Path(resume_from)
        if resume_path.exists():
            prior = json.loads(resume_path.read_text(encoding="utf-8"))
            results = prior.get("results", [])

    # Report path 
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

    # Thread-safe shared state
    results_lock = asyncio.Lock()

    def _write_checkpoint():
        """Write current results to disk. Called inside the lock."""
        report_data: Dict[str, Any] = {
            "metadata": {
                "eval_mode":           "dynamic",
                "personas_file":       personas_file,
                "rubric_file":         rubric_file,
                "rubric_type":         rubric.get("rubric_type", "response"),
                "tutor_type":          tutor_type,
                "student_model":       STUDENT_MODEL,
                "judge_model":         JUDGE_MODEL,
                "max_turns_per_case":  max_turns_per_case,
                "concurrency":         concurrency,
                "versions": {
                    "base_srl":    VERSION_BASE_SRL,
                    "router":      VERSION_ROUTER,
                    "chain":       VERSION_CHAIN,
                    "respond":     VERSION_RESPOND,
                    "forethought": VERSION_FORETHOUGHT,
                    "performance": VERSION_PERFORMANCE,
                    "reflection":  VERSION_REFLECTION,
                },
                "completed_cases":       len(results),
                "failed_cases":          len(failed_cases),
                "total_cases_requested": len(personas),
            },
            "results":  results,
            "failures": failed_cases,
        }
        report_path.write_text(
            json.dumps(report_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    sem = asyncio.Semaphore(concurrency)

    async def _run_one(idx: int, persona: Dict[str, Any]) -> None:
        """Run one persona, collect result or failure, write checkpoint."""
        async with sem:
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
                async with results_lock:
                    results.append(result)
                    _write_checkpoint()

            except Exception as e:
                case_id = persona.get("id", f"persona_{idx}")
                if verbose:
                    print(
                        f"[DYN] ERROR case {idx}/{len(personas)}: "
                        f"{case_id} -> {type(e).__name__}: {e}"
                    )
                async with results_lock:
                    failed_cases.append({
                        "case_id":       case_id,
                        "persona_index": idx,
                        "error_type":    type(e).__name__,
                        "error":         str(e),
                        "persona":       persona,
                    })
                    _write_checkpoint()

    # Run all cases concurrently 
    # Offset index by the number of already-completed cases so persona_idx
    # numbers remain meaningful relative to the full original list.
    start_idx = len(completed_ids) + 1
    await asyncio.gather(*[
        _run_one(start_idx + i, persona)
        for i, persona in enumerate(personas_to_run)
    ])

    if verbose:
        async with results_lock:
            n_ok   = len(results)
            n_fail = len(failed_cases)
        print(
            f"\n[DYN] Finished.  "
            f"completed={n_ok}  failed={n_fail}  "
            f"total={len(personas)}"
        )
        print(f"[DYN] Report: {report_path}")
    return str(report_path)


# CLI
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
        help="Rubric YAML filename inside evaluator/rubrics/",
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
        help="Maximum turns per conversation (default: 3).",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY,
        metavar="N",
        help=f"Number of cases to run simultaneously (default: {CONCURRENCY}).",
    )
    p.add_argument(
        "--resume-from",
        default=None,
        metavar="PATH",
        help=(
            "Path to a prior checkpoint report.  "
            "Already-completed cases will be skipped."
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
            concurrency=args.concurrency,
            verbose=not args.quiet,
            resume_from=args.resume_from,
        )
    )