"""
DriveGuard — Server Startup & Orchestration Runner
===================================================
Final-Year Computer Engineering Project:
Personalised Adaptive AI Driver Drowsiness Detection and Early Warning System
Using Computer Vision.

Usage:
    python run.py
    python run.py --port 8000 --host 127.0.0.1 --reload
    python run.py --check-only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
VENV_PYTHON_WIN = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
VENV_PYTHON_POSIX = ROOT_DIR / ".venv" / "bin" / "python"


def is_running_in_project_venv() -> bool:
    """Check whether the active sys.executable is inside the local .venv."""
    try:
        active_exec = Path(sys.executable).resolve()
        venv_dir = (ROOT_DIR / ".venv").resolve()
        return venv_dir in active_exec.parents or active_exec.is_relative_to(venv_dir)
    except (ValueError, AttributeError):
        return str(ROOT_DIR / ".venv") in str(sys.executable)


def auto_switch_to_venv_if_needed() -> None:
    """If invoked by an external/global Python, relaunch using .venv interpreter."""
    if is_running_in_project_venv():
        return

    target_python = VENV_PYTHON_WIN if os.name == "nt" else VENV_PYTHON_POSIX
    if target_python.is_file():
        print(f"[DriveGuard Runner] Detected outside virtual environment ({sys.executable}).")
        print(f"[DriveGuard Runner] Auto-switching to project venv: {target_python}\n")
        cmd = [str(target_python), str(Path(__file__).resolve())] + sys.argv[1:]
        ret = subprocess.call(cmd)
        sys.exit(ret)
    else:
        print("[DriveGuard Runner WARNING] Local .venv not found. Proceeding with current interpreter...")


def run_preflight_checks() -> bool:
    """Verify models, directories, and core dependencies before launching server."""
    # Ensure UTF-8 output encoding on Windows terminals if supported
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 72)
    print("   [DriveGuard] Personalised Adaptive AI Driver Monitoring System")
    print("=" * 72)
    print(f"Python:       {sys.version.split()[0]} ({sys.executable})")
    print(f"Project Root: {ROOT_DIR}")

    # 1. Directory Checks
    weights_dir = ROOT_DIR / "backend" / "weights"
    profiles_dir = ROOT_DIR / "backend" / "research" / "outputs" / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    # 2. Weights Checks
    landmarker_task = weights_dir / "face_landmarker.task"
    eye_onnx = weights_dir / "mobilenetv3_eye.onnx"

    missing = []
    if not landmarker_task.is_file():
        missing.append(f"MediaPipe Landmarker Task: {landmarker_task}")
    if not eye_onnx.is_file():
        missing.append(f"MobileNetV3 ONNX Model:    {eye_onnx}")

    if missing:
        print("\n[!] MISSING CRITICAL WEIGHTS:")
        for item in missing:
            print(f"   - {item}")
        return False

    print(f"Weights:      MediaPipe Landmarker (OK) | MobileNetV3-Small ONNX (OK)")

    # 3. Dependency Imports
    try:
        import fastapi
        import uvicorn
        import mediapipe
        import onnxruntime
        import cv2
        import psutil
        extra_info = []
        try:
            import torch
            extra_info.append(f"PyTorch v{torch.__version__}")
        except ImportError:
            extra_info.append("PyTorch (optional/offline)")
        print(f"Dependencies: FastAPI v{fastapi.__version__} | MediaPipe v{mediapipe.__version__} | ONNXRuntime v{onnxruntime.__version__} | {' | '.join(extra_info)}")
    except ImportError as exc:
        print(f"\n[!] DEPENDENCY ERROR: Missing package: {exc}")
        print("Please run: pip install -r backend/requirements.txt")
        return False

    print("Preflight:    All checks PASSED [OK]")
    print("=" * 72)
    return True


def main() -> None:
    auto_switch_to_venv_if_needed()

    parser = argparse.ArgumentParser(
        description="DriveGuard FastAPI Backend & Telemetry Server Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default=os.getenv("HOST", "0.0.0.0"), help="Host interface to bind to")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")), help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn hot-reload")
    parser.add_argument("--check-only", action="store_true", help="Run preflight checks and exit")
    args = parser.parse_args()

    # Ensure project root is in sys.path
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))

    passed = run_preflight_checks()
    if not passed:
        sys.exit(1)

    if args.check_only:
        print("Preflight check completed successfully.")
        sys.exit(0)

    import uvicorn

    print(f"\n[DriveGuard] Starting FastAPI Server...")
    print(f"   - REST API:           http://{args.host}:{args.port}")
    print(f"   - Interactive Docs:   http://{args.host}:{args.port}/docs")
    print(f"   - Health Endpoint:    http://{args.host}:{args.port}/health")
    print(f"   - WebSocket Stream:   ws://{args.host}:{args.port}/ws/v1/session/{{session_id}}")
    print(f"\nPress Ctrl+C to stop the server.\n")

    try:
        uvicorn.run(
            "backend.app.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=1,
            log_level="info",
        )
    except KeyboardInterrupt:
        print("\n[DriveGuard] Server stopped by user.")


if __name__ == "__main__":
    main()
