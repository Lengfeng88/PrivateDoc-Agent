# ─────────────────────────────────────────────────────────────────────────────
# PrivateDoc Agent — Dockerfile
# Multi-stage build:
#   base     → Python + system deps (shared)
#   builder  → pip install into /install (no cache in runtime)
#   runtime  → minimal image, non-root user, ~350 MB
# ─────────────────────────────────────────────────────────────────────────────

# ── base ─────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── builder ───────────────────────────────────────────────────────────────────
FROM base AS builder

COPY pyproject.toml ./
# Generate a minimal requirements.txt if uv/pip-compile is not available
RUN pip install --upgrade pip && \
    pip install \
        fastapi==0.111.* \
        uvicorn[standard]==0.30.* \
        qdrant-client==1.10.* \
        langchain==0.2.* \
        langchain-community==0.2.* \
        langgraph==0.1.* \
        openai==1.40.* \
        sentence-transformers==3.0.* \
        torch==2.3.* --index-url https://download.pytorch.org/whl/cpu \
        spacy==3.7.* \
        unstructured[pdf,docx]==0.14.* \
        python-multipart==0.0.9 \
        pydantic==2.8.* \
        pydantic-settings==2.3.* \
        httpx==0.27.* \
        tenacity==8.4.* \
        prometheus-fastapi-instrumentator==7.0.* \
        ragas==0.1.* \
        rank-bm25==0.2.* \
        --target /install

# Download spaCy model into the install dir
RUN python -m spacy download en_core_web_sm --target /install 2>/dev/null || true

# ── runtime ───────────────────────────────────────────────────────────────────
FROM base AS runtime

# Copy installed packages
COPY --from=builder /install /usr/local/lib/python3.11/site-packages

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Copy application source
COPY api/      ./api/
COPY agent/    ./agent/
COPY ingest/   ./ingest/
COPY router/   ./router/
COPY eval/     ./eval/

# Writable log dir owned by appuser
RUN mkdir -p /app/logs && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Healthcheck baked into image
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
