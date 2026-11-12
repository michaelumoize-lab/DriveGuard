# DriveGuard — Convenience Launchers

This directory contains one-click executable shortcuts to launch DriveGuard subsystems without needing to type long commands manually.

You can run them in **two ways**:
1. **Double-click** any `.bat` file directly in Windows File Explorer.
2. **Execute from terminal** (PowerShell or Command Prompt).

---

## Available Launchers

| File | What It Runs | Shortcut Purpose |
| :--- | :--- | :--- |
| `01_run_phone_camera.bat` | `python scripts/run_camera_test.py --url <phone_url>` | Streams live video from your Android/iPhone |
| `02_run_laptop_webcam.bat` | `python scripts/run_camera_test.py --device 0` | Streams from built-in HP EliteBook webcam |
| `03_run_unit_tests.bat` | `pytest -v` | Runs the full automated verification test suite |
| `04_probe_cameras.bat` | `python scripts/run_camera_test.py --probe` | Probes and lists all local video capture indices |

---

## Configuring the Phone Camera Stream URL

`01_run_phone_camera.bat` and `01_run_phone_camera.ps1` automatically resolve the phone video stream URL in the following priority order:

1. **CLI Parameter**: Pass the URL directly when running from command line:
   ```cmd
   launchers\01_run_phone_camera.bat http://192.168.1.50:8080/video
   ```
2. **Environment Variable**: Set `DRIVEGUARD_PHONE_URL`:
   ```cmd
   set DRIVEGUARD_PHONE_URL=http://192.168.1.50:8080/video
   ```
3. **Configuration File**: Saved in `launchers/phone_url.txt`. You can edit this file directly to set your phone's Wi-Fi IP and port.
4. **Interactive Prompt**: When double-clicking the batch file, it prints the configured URL. Press **ENTER** to connect immediately, or type a new URL to connect and save it automatically for next time.

---

## Keyboard Controls While Previewing

- **`m`** : Toggle horizontal mirror flip on/off live in real time.
- **`s`** : Save a diagnostic snapshot PNG to disk.
- **`q`** or **`ESC`** : Exit cleanly and print performance summary.
