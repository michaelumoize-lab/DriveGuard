# DriveGuard — Convenience Launchers

This directory contains one-click executable shortcuts to launch DriveGuard subsystems without needing to type long commands manually.

You can run them in **two ways**:
1. **Double-click** any `.bat` file directly in Windows File Explorer.
2. **Execute from terminal** (PowerShell or Command Prompt).

---

## Available Launchers

| File | What It Runs | Shortcut Purpose |
| :--- | :--- | :--- |
| `01_run_phone_camera.bat` | `python scripts/run_camera_test.py --url http://192.168.0.103:8080/video` | Streams live video from your Android/iPhone |
| `02_run_laptop_webcam.bat` | `python scripts/run_camera_test.py --device 0` | Streams from built-in HP EliteBook webcam |
| `03_run_unit_tests.bat` | `pytest -v` | Runs the full automated verification test suite |
| `04_probe_cameras.bat` | `python scripts/run_camera_test.py --probe` | Probes and lists all local video capture indices |

---

## Keyboard Controls While Previewing

- **`m`** : Toggle horizontal mirror flip on/off live in real time.
- **`s`** : Save a diagnostic snapshot PNG to disk.
- **`q`** or **`ESC`** : Exit cleanly and print performance summary.
