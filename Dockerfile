FROM python:3.11-slim

# Install system graphics, EGL, and OpenGL runtime dependencies required by MediaPipe & OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 \
    libgles2 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend codebase and weights
COPY . .

# Expose Render standard port (defaults to 10000 on Render)
ENV PORT=10000
EXPOSE 10000

# Start DriveGuard FastAPI via Uvicorn
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
