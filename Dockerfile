# RAG Evaluation Platform — multi-stage production image
# Stage 1: build wheel + install deps
# Stage 2: slim runtime with non-root user

FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app ./app
COPY frontend ./frontend
COPY datasets ./datasets

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# ---------------------------------------------------------------------------

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    APP_DEBUG=false \
    HOST=0.0.0.0 \
    PORT=8000 \
    MCP_PORT=8100 \
    CHROMA_PERSIST_DIRECTORY=/data/chroma \
    REPORT_DIRECTORY=/data/reports \
    BENCHMARK_DIRECTORY=/data/benchmarks \
    QUALITY_REPORT_DIRECTORY=/data/quality_reports \
    BENCHMARK_HISTORY_PATH=/data/benchmarks/history.json \
    EMPLOYEE_DATA_PATH=/data/employees/employees.json

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 1000 rag \
    && useradd --system --uid 1000 --gid rag --home /app --shell /usr/sbin/nologin rag \
    && mkdir -p /data/chroma /data/reports /data/benchmarks /data/quality_reports /data/employees \
    && chown -R rag:rag /app /data

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY app ./app
COPY frontend ./frontend
COPY datasets ./datasets
COPY scripts ./scripts
COPY pyproject.toml README.md ./

USER rag

EXPOSE 8000 8100 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host ${HOST} --port ${PORT}"]
