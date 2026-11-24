# =============================================================================
# Multi-stage Dockerfile for the BTC forecast API
#
# Build from the REPO ROOT (not from inside app/):
#   docker build -f app/Dockerfile -t btc-forecast-api:latest .
#
# Stage 1 "builder": installs Python libraries into a virtual environment.
# Stage 2 "seed":    downloads BTC data and trains a first model, so the image
#                    can serve predictions the moment it starts.
# Stage 3 "runtime": small final image. Copies only the venv, code and model.
#                    Compilers, caches and training-only libraries are left behind.
# =============================================================================

# ---------- Stage 1: install dependencies -----------------------------------
FROM python:3.10-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Copy requirements first: Docker caches this layer, so code changes do not
# trigger a slow reinstall of every library.
COPY app/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# ---------- Stage 2: train a seed model -------------------------------------
FROM builder AS seed
RUN pip install "yfinance>=0.2.54" "requests>=2.31"
WORKDIR /build
COPY src/ ./src/
ENV DATA_DIR=/build/data MODEL_DIR=/build/models
RUN python src/etl_fetch.py \
 && python src/features.py \
 && python src/train_forecast.py

# ---------- Stage 3: final runtime image ------------------------------------
FROM python:3.10-slim AS runtime

LABEL org.opencontainers.image.title="btc-forecast-api" \
      org.opencontainers.image.description="7-day Bitcoin close price forecast API"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models \
    PORT=5000

# Security: run as a normal user, never as root.
RUN groupadd --gid 1000 appuser \
 && useradd --uid 1000 --gid appuser --no-create-home --shell /usr/sbin/nologin appuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser src/features.py src/config.py src/__init__.py ./src/
COPY --chown=appuser:appuser app/app.py app/entrypoint.sh ./
COPY --from=seed --chown=appuser:appuser /build/models/model.joblib /build/models/metrics.json ./seed_models/

RUN mkdir -p /app/models && chown appuser:appuser /app/models && chmod +x /app/entrypoint.sh

USER appuser
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,os; urllib.request.urlopen('http://localhost:'+os.environ.get('PORT','5000')+'/health')" || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
