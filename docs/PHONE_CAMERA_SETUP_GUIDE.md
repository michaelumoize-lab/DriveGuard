# DriveGuard — Phone Camera Integration & Setup Guide (Phase 2)

**Project:** DriveGuard — Personalized AI-Based Driver Drowsiness Detection and Early Warning System  
**Document Version:** 1.0 (Phase 2 Milestone)  
**Audience:** Developer, Vehicle Operators, Academic Supervisor  
**Supported Devices:** Android (e.g., Xiaomi Redmi 13C) and iOS (e.g., Apple iPhone 16)  

---

## 1. Motivation: Why Use a Phone Camera for Driver Monitoring?

Modern smartphones (such as the iPhone 16 and Redmi 13C) possess optical sensor hardware that often far exceeds laptop webcams:
- **Superior Low-Light Sensitivity:** Larger sensor pixels (f/1.6 to f/2.2 apertures) yield significantly higher signal-to-noise ratio (SNR) in dark car cabins.
- **Wide Dynamic Range (HDR):** Handles sudden headlights, streetlights, and direct sunlight glare better than standard 720p laptop webcams.
- **Flexible Cabin Mounting:** A phone can be placed on a windshield suction mount or A-pillar dashboard mount directly at driver eye level (60–80 cm), avoiding the awkward steep upward angle of a center-console laptop.

DriveGuard provides a dedicated `PhoneCamera` adapter that ingests these video streams over a local connection with **zero cloud exposure**.

---

## 2. Supported Transport Architectures

DriveGuard supports two distinct local transport methods:

```text
┌─────────────────────────────────────────────────────────────┐
│ OPTION A: USB Virtual DirectShow (Zero Wi-Fi Jitter)        │
│ Phone Camera ──(USB Tether)──> PC DirectShow Device (e.g. 1) │
│ Handled natively by: OpenCVCamera(device_index=1)            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ OPTION B: Local Network IP Stream (Zero Software on PC)     │
│ Phone Camera ──(Local Wi-Fi)──> http://<phone-ip>:8080/video│
│ Handled natively by: PhoneCamera(stream_url="http://...")    │
└─────────────────────────────────────────────────────────────┘
```

Both options feed into the same `BaseCamera` interface and produce identical `FramePacket` containers.

---

## 3. Step-by-Step Setup: Option B (Local Wi-Fi / IP Stream)

This approach requires no proprietary PC drivers. Your phone and laptop simply need to be connected to the **same local Wi-Fi network** (or the laptop can connect to the phone's personal Wi-Fi Hotspot).

### For Android Phones (e.g. Xiaomi Redmi 13C)

1. **Install App:** Download **IP Webcam** by Pavel Khlebovich from the Google Play Store (free, open, privacy-respecting).
2. **Configure Settings:**
   - **Video Resolution:** Select `1280x720` (720p HD is optimal for MediaPipe inference without thermal throttling).
   - **Quality:** Set to `60%–70%` (balances visual sharpness with low Wi-Fi latency).
   - **Orientation:** Lock to **Landscape** (or portrait depending on your vehicle mount).
3. **Start Server:** Scroll to the bottom and tap **"Start server"**.
4. **Read Stream URL:** The phone screen will display an address, typically:
   ```text
   http://192.168.1.50:8080
   ```
5. **Video Feed Endpoint:** The direct video feed URL for DriveGuard is:
   ```text
   http://192.168.1.50:8080/video
   ```

---

### For iOS Phones (e.g. Apple iPhone 16)

1. **Install App:** Download **IP Camera Lite** or **Live-Reporter** or **DroidCam** from the iOS App Store.
2. **Configure Settings:**
   - Set resolution to `1280x720` @ `30 FPS`.
   - Disable audio transmission to save Wi-Fi bandwidth.
3. **Start Broadcast:** Start the local broadcast server. The app displays your local stream URL, for example:
   ```text
   http://192.168.1.65:8080/video
   # Or for RTSP:
   rtsp://192.168.1.65:554/live
   ```

---

## 4. How to Run the Phone Stream in DriveGuard

Once the phone is broadcasting, test the connection using the DriveGuard benchmark runner:

```powershell
# In your project root with .venv311 active:
& ".\.venv311\Scripts\python.exe" scripts/run_camera_test.py --url http://<phone-ip>:8080/video
```

### Example:
```powershell
& ".\.venv311\Scripts\python.exe" scripts/run_camera_test.py --url http://192.168.1.50:8080/video
```

### What Happens:
1. `create_camera()` automatically detects the `http://` prefix and instantiates `PhoneCamera`.
2. A dedicated capture thread connects to the phone and flushes old network packets.
3. The OpenCV GUI opens with the live phone camera feed.
4. The real-time HUD displays:
   - **Source:** `phone:http://192.168.1.50:8080/video`
   - **Ingestion FPS:** True frame rate coming over Wi-Fi.
   - **Frame Latency:** Measured age in milliseconds (typically 15–35 ms on 5GHz Wi-Fi).
   - **Dropped Frames:** Number of unread frames dropped to preserve real-time safety.

---

## 5. In-Vehicle Physical Mounting & Thermal Management

When deploying in a real vehicle:

1. **Mounting Position:**
   - Use a sturdy magnetic or clamp windshield/dashboard mount positioned on the driver's side of the center console or lower A-pillar.
   - Ensure the lens is angled directly toward the driver's eyes at a distance of **60 to 90 cm**.
2. **Thermal Mitigation:**
   - Continuous camera streaming generates internal heat. Avoid direct exposure to windshield defroster vents.
   - Keep phone screen brightness at minimum (or enable the "turn screen off while streaming" feature in the IP Webcam app).
3. **Power:**
   - Connect the phone to a 5V/2A or USB-PD car charger. Do not rely solely on phone battery for extended monitoring sessions.
4. **Notifications:**
   - Enable **"Do Not Disturb"** or **"Driving Focus"** on the phone to prevent incoming phone calls from suspending the camera feed.

---

## 6. Academic & Viva Defense Points

### Q: Does streaming video from the phone over Wi-Fi compromise driver privacy?
**A:** No. DriveGuard uses a point-to-point local subnet connection (either home/car Wi-Fi or a direct Wi-Fi Hotspot between the phone and laptop). Zero bytes of video data are sent over the public internet or through any cloud servers.

### Q: What happens if the phone's Wi-Fi connection drops or jitters?
**A:**
1. **Zero Buffer Bloat:** The `FrameBuffer` holds exactly 1 frame (`capacity=1`). If Wi-Fi stalls for 200 ms and then delivers a burst of frames, older frames are instantly dropped, ensuring the driver is never alerted based on outdated footage.
2. **Watchdog Reconnection:** The `PhoneCamera` adapter runs a hardware watchdog. If no frame arrives within `timeout_sec` (default: 3.0s), the system marks the feed as disconnected and initiates an automatic reconnection loop without crashing the application.
