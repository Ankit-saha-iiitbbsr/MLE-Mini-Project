# =============================================================================
# DefectVision inference service (M4)
#
# Multi-stage: the build stage carries pip, compilers and wheel caches; the
# runtime stage receives only the installed virtualenv and the application.
# That keeps the shipped image free of build tooling -- smaller to pull and a
# smaller attack surface.
#
#   docker build -t defectvision:1.0.0 .
#   docker run --rm -p 8000:8000 \
#       -v "$(pwd)/models/production:/app/models/production:ro" \
#       -v "$(pwd)/monitoring:/app/monitoring" \
#       defectvision:1.0.0
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1 - build
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Torch first, from the CPU index. On Linux the default PyPI wheel pulls the
# full CUDA stack (~2.5 GB) which this image will never use.
RUN pip install --upgrade pip setuptools wheel && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2" "torchvision>=0.17"

# Serving needs a fraction of the training stack. MLflow and DVC are
# deliberately absent: experiment tracking and data versioning are development
# concerns, and adding them here would put ~200 MB and a database driver on the
# request path for no runtime benefit.
RUN pip install \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.29" \
        "python-multipart>=0.0.9" \
        "pydantic>=2.6" \
        "pydantic-settings>=2.2" \
        "numpy>=1.26,<3.0" \
        "pandas>=2.1" \
        "scipy>=1.11" \
        "Pillow>=10.2" \
        "PyYAML>=6.0" \
        "typer>=0.12" \
        "rich>=13.7"

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-deps .

# -----------------------------------------------------------------------------
# Stage 2 - runtime
# -----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="DefectVision" \
      org.opencontainers.image.description="Image-based casting defect / quality classifier" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.source="https://github.com/<your-org>/ML-Engg-mini-project"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEFECTVISION_LOG_JSON=1 \
    # Torch spawns one thread per core by default; inside a container that
    # oversubscribes the CPU quota and makes p99 latency worse, not better.
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

# libgomp is required by torch; curl is used by the HEALTHCHECK below.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Never run the service as root.
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser params.yaml ./
COPY --chown=appuser:appuser src/ ./src/

# Mount points for the promoted model (read-only) and the prediction log.
RUN mkdir -p /app/models/production /app/monitoring /app/data \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

# Readiness, not liveness: a container whose bundle failed to load is running
# but must not be sent traffic.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/readyz || exit 1

CMD ["uvicorn", "defectvision.serving.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
