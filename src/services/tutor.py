from src.services.prompt_loader import load_prompt

TUTOR_MODEL = "gpt-4o-mini"

PHASE_TO_FILE = {
    "FORETHOUGHT": "phases/forethought.txt",
    "PERFORMANCE": "phases/performance.txt",
    "REFLECTION": "phases/reflection.txt",
}

STRATEGY_TO_FILE = {
    "NONE": None,
    # "HINT_LADDER": "strategies/hint_ladder.txt",
    # "DEBUG_COACH": "strategies/debug_coach.txt",
    # "SOCRATIC": "strategies/socratic.txt",
    # "EXAM_SAFE": "strategies/exam_safe.txt",
}

def build_system_prompt(tutor_type: str, route: dict) -> str:
    # Base prompt
    if tutor_type == "SRL Tutor":
        base = load_prompt("base/simple_base_SRL_v0.txt")
    else:
        base = load_prompt("base/ai_base_control.txt")

    # Phase prompt (only for SRL tutor)
    phase_prompt = ""
    if tutor_type == "SRL Tutor":
        phase = route.get("phase", "PERFORMANCE")
        phase_file = PHASE_TO_FILE.get(phase, PHASE_TO_FILE["PERFORMANCE"])
        phase_prompt = load_prompt(phase_file)

    # Strategy prompt
    strategy_prompt = ""
    strategy = route.get("strategy", "NONE")
    strat_file = STRATEGY_TO_FILE.get(strategy)
    if strat_file:
        strategy_prompt = load_prompt(strat_file)

    parts = [base, phase_prompt, strategy_prompt]
    return "\n\n".join([p for p in parts if p and p.strip()])


async def run_tutor(client, system_prompt: str, llm_history: list) -> str:
    resp = await client.chat.completions.create(
        model=TUTOR_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            *llm_history
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content or ""