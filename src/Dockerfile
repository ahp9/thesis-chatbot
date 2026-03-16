# Label this first part as "base"
FROM python:3.11-slim AS base

WORKDIR /app

RUN python -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── dev: chainlit web app ─────────────────────────────────────────────────────
FROM base AS dev

COPY . .

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]

# ── judge: evaluation runner ──────────────────────────────────────────────────
FROM base AS judge
 
CMD ["python", "-m", "evaluator.evaluator_v1"]