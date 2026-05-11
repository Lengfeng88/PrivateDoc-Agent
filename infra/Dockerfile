FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

FROM base AS builder

COPY pyproject.toml ./

RUN pip install --upgrade pip && \
    pip install \
        "fastapi>=0.111,<0.113" \
        "uvicorn[standard]>=0.30,<0.31" \
        "qdrant-client>=1.10,<1.12" \
        "langchain>=0.2,<0.3" \
        "langchain-community>=0.2,<0.3" \
        "langgraph>=0.1,<0.2" \
        "openai>=1.40,<2" \
        "sentence-transformers>=3.0,<4" \
        "spacy>=3.7,<4" \
        "unstructured[pdf,docx]>=0.14,<0.15" \
        "python-multipart>=0.0.9" \
        "pydantic>=2.8,<3" \
        "pydantic-settings>=2.3,<3" \
        "httpx>=0.27,<0.29" \
        "tenacity>=8.4,<9" \
        "ragas>=0.1,<0.2" \
        "rank-bm25>=0.2,<0.3" \
        "loguru>=0.7" \
        "sse-starlette>=1.8" \
        "starlette>=0.37,<0.38" \
        --target /install && \
    pip install "torch>=2.3" \
        --index-url https://download.pytorch.org/whl/cpu \
        --target /install

FROM base AS runtime

COPY --from=builder /install /usr/local/lib/python3.11/site-packages

RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

COPY api/      ./api/
COPY agent/    ./agent/
COPY ingest/   ./ingest/
COPY router/   ./router/
COPY eval/     ./eval/

RUN mkdir -p /app/logs && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
