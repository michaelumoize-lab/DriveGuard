"""
DriveGuard — FastAPI Production Application Entrypoint.

Provides:
  - Async Lifespan model preloading for 0-latency first frame response.
  - High-throughput Binary WebSocket (`/ws/v1/session/{session_id}`) for 30 FPS client video streams.
  - RESTful APIs (`/api/v1`) for driver profiles, session analytics, and calibration wizard.
  - OpenAPI Swagger documentation at `/docs`.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any
import traceback

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes.calibration import router as calibration_router
from backend.app.api.routes.profiles import router as profiles_router
from backend.app.api.routes.sessions import router as sessions_router
from backend.app.api.ws_telemetry import ws_router
from backend.app.core.config import settings
from backend.app.services.session_manager import session_manager

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Preload AI models into GPU/CPU memory on server startup."""
    print("[Lifespan] Preloading MediaPipe Face Landmarker & ONNX Runtime models...")
    try:
        session_manager.preload_shared_models()
        print("[Lifespan] Shared AI models successfully loaded and ready for inference.")
    except Exception as exc:
        print(f"[Lifespan WARNING] Lazy model initialization will be used: {exc}")
    yield
    print("[Lifespan] Shutting down DriveGuard AI Service.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Personalised Adaptive AI Driver Drowsiness Detection & Early Warning System. "
        "Provides 4-Zone Pose-Conditioned Calibration, Reliability-Weighted Confidence-Gated "
        "Cascade, and Explainable Hysteresis Risk Scoring."
    ),
    lifespan=lifespan,
)

# CORS Middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Capture any uncaught runtime exception and return structured JSON with traceback."""
    tb = traceback.format_exc()
    print(f"[ERROR] Uncaught exception on {request.method} {request.url.path}: {exc}\n{tb}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "ERROR",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": tb,
        },
    )

# Root & Health Check Endpoints
@app.get("/", tags=["Status"])
def root_info() -> Dict[str, str]:
    """Service status and OpenAPI documentation pointer."""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "RUNNING",
        "docs_url": "/docs",
        "websocket_url": "/ws/v1/session/{session_id}",
    }


@app.get("/health", tags=["Status"])
def health_check() -> Dict[str, object]:
    """Diagnostic health check endpoint."""
    uptime_sec = round(time.time() - START_TIME, 1)
    active_sessions = len(session_manager.list_active_sessions())
    weights_onnx_present = (settings.WEIGHTS_DIR / "mobilenetv3_eye.onnx").is_file()
    weights_task_present = (settings.WEIGHTS_DIR / "face_landmarker.task").is_file()

    return {
        "status": "HEALTHY",
        "uptime_sec": uptime_sec,
        "active_sessions": active_sessions,
        "weights": {
            "mobilenetv3_eye_onnx": weights_onnx_present,
            "face_landmarker_task": weights_task_present,
        },
    }


# Mount REST API Routes under /api/v1
api_v1_router = APIRouter(prefix=settings.API_V1_STR)
api_v1_router.include_router(sessions_router)
api_v1_router.include_router(profiles_router)
api_v1_router.include_router(calibration_router)

app.include_router(api_v1_router)

# Mount Binary WebSocket Endpoint
app.include_router(ws_router)
