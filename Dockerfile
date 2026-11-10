# ==============================================================================
# DriveGuard — Production AI Inference Microservice Multi-Stage Dockerfile
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build & Dependency Resolution
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools if wheels require compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

# ------------------------------------------------------------------------------
# Stage 2: Minimal Production Runtime
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runner

# Install runtime shared libraries required by OpenCV, MediaPipe, and GLib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    libgles2 \
    libglib2.0-0 \
    libgomp1 \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy backend application, weights, and orchestration runner
COPY backend/app ./backend/app
COPY backend/weights ./backend/weights
COPY backend/data ./backend/data
COPY run.py .
COPY requirements.txt .

# Create non-root system user for security compliance
RUN useradd -m -u 10001 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose Render / standard cloud container port (defaults to 10000 on Render)
ENV PORT=10000 \
    HOST=0.0.0.0 \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

EXPOSE 10000

# Container Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/health || exit 1

# Start DriveGuard FastAPI via Uvicorn
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
