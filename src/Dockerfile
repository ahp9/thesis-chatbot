FROM python:3.11-slim AS base

# Prevents Python from writing .pyc files and buffers logs nicely for Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
  && rm -rf /var/lib/apt/lists/*

# This will now look for requirements.txt in the build context (Root)
COPY requirements.txt .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Dev stage ----------
FROM base AS dev
EXPOSE 8000
# Note: In dev, we usually mount the src folder via docker-compose
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]

# ---------- Prod stage ----------
FROM base AS prod
# Copy everything from the src folder into /app
COPY src/ /app/ 
EXPOSE 8000
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]