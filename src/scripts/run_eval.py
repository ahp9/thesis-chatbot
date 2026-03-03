import asyncio

from src.evaluator.evaluator import run_suite

async def main():
    path = await run_suite(
        suite_file="forethought_tricky_v1.json",
        rubric_file="forethought_v1.yaml",
        tutor_type="SRL Tutor"
    )
    print("Wrote:", path)

asyncio.run(main())