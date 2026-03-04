import asyncio

from evaluator.evaluator import run_suite

async def main():
    path = await run_suite(
        suite_file="performance_convo.json",
        rubric_file="performance_phase.yaml",
        tutor_type="SRL Tutor"
    )
    print("Wrote:", path)

asyncio.run(main())