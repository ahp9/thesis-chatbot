FROM python:3.11-slim AS base

# Prevents Python from writing .pyc files and buffers logs nicely for Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt


# ---------- Dev stage ----------
FROM base AS dev

# Optional dev tools (linters, reload tools, etc.)
# RUN pip install --no-cache-dir ruff

EXPOSE 8000

# In dev we run directly.
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]


# ---------- Prod stage ----------
FROM base AS prod

COPY . /app

EXPOSE 8000

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
