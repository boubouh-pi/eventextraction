# Event Extraction API — production image
FROM python:3.12-slim

# libgomp1: OpenMP runtime required by LightGBM; curl: container healthcheck
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Install dependencies first for layer caching
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

# Application code + trained artifacts (models/ is committed, ~200KB)
COPY app/ app/
COPY src/dateclf/ src/dateclf/
COPY models/ models/

# Non-root user
RUN useradd --create-home apiuser \
    && mkdir -p /srv/data/monitoring \
    && chown -R apiuser:apiuser /srv
USER apiuser

ENV PORT=8000 \
    MONITORING_PATH=/srv/data/monitoring/requests.jsonl \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
