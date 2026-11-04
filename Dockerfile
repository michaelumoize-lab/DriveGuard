# DriveGuard Production Dockerfile for Render Cloud Deployment
FROM python:3.11-slim

# Prevent python from buffering stdout/stderr and creating pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install system dependencies required for image/video decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definition and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and pre-trained weights
COPY backend/ /app/backend/
COPY run.py /app/run.py

# Render dynamic port exposure
EXPOSE 8000

# Diagnostic container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Launch ASGI server with uvicorn bound to 0.0.0.0 and dynamic Render port
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
