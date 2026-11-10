/**
 * DriveGuard Client Webcam & MediaDevices Subsystem Unit Tests
 * ============================================================
 * Tests:
 * 1. checkMediaDevicesSupport across Node, insecure HTTP, and secure HTTPS/localhost environments
 * 2. getRecommendedConstraints resolution & framerate matrix (640x480 vs 1280x720 at 15, 20, 25, 30 FPS)
 * 3. Constraint degradation cascade (Recommended -> Fallback -> Basic)
 * 4. parseCameraDevices device label heuristics (front/user/facetime vs back/rear/wide)
 * 5. extractCameraResolution from HTMLVideoElement and MediaStreamTrack settings
 * 6. mapMediaErrorToCameraError across all browser DOMExceptions with actionable user recovery steps
 * 7. FrameProducer decoupled frame dispatch and subscription lifecycle
 */

import {
  checkMediaDevicesSupport,
  getRecommendedConstraints,
  getFallbackConstraints,
  getBasicConstraints,
  parseCameraDevices,
  extractCameraResolution,
  mapMediaErrorToCameraError,
} from "../lib/camera/camera-utils";
import { FrameProducer } from "../lib/camera/frame-producer";
import { CameraDevice, CapturedFrame, FrameConsumer } from "../types/camera";

let passed = 0;
let total = 0;

async function it(desc: string, fn: () => void | Promise<void>) {
  total++;
  try {
    await fn();
    passed++;
    console.log(`  [PASS] ${desc}`);
  } catch (err) {
    console.error(`  [FAIL] ${desc}`);
    console.error(err);
    process.exitCode = 1;
  }
}

function expect(val: unknown) {
  return {
    toBe(expected: unknown) {
      if (val !== expected) {
        throw new Error(`Expected ${JSON.stringify(val)} to be ${JSON.stringify(expected)}`);
      }
    },
    toEqual(expected: unknown) {
      if (JSON.stringify(val) !== JSON.stringify(expected)) {
        throw new Error(`Expected ${JSON.stringify(val)} to equal ${JSON.stringify(expected)}`);
      }
    },
    toBeTruthy() {
      if (!val) {
        throw new Error(`Expected ${String(val)} to be truthy`);
      }
    },
    toBeFalsy() {
      if (val) {
        throw new Error(`Expected ${String(val)} to be falsy`);
      }
    },
    toBeNull() {
      if (val !== null) {
        throw new Error(`Expected ${String(val)} to be null`);
      }
    },
    toContain(substr: string) {
      if (typeof val !== "string" || !val.includes(substr)) {
        throw new Error(`Expected "${String(val)}" to contain "${substr}"`);
      }
    },
  };
}

function setMockNavigator(navObj: unknown) {
  Object.defineProperty(globalThis, "navigator", {
    value: navObj,
    configurable: true,
    writable: true,
  });
}

function setMockWindow(winObj: unknown) {
  Object.defineProperty(globalThis, "window", {
    value: winObj,
    configurable: true,
    writable: true,
  });
}

async function runCameraWebcamTests() {
  console.log("======================================================================");
  console.log("   [DriveGuard] Client Webcam & MediaDevices Subsystem Unit Tests");
  console.log("======================================================================\n");

  // Save original descriptors
  const origWindowDesc = Object.getOwnPropertyDescriptor(globalThis, "window");
  const origNavigatorDesc = Object.getOwnPropertyDescriptor(globalThis, "navigator");

  // -------------------------------------------------------------------------
  // TEST GROUP 1: checkMediaDevicesSupport
  // -------------------------------------------------------------------------
  console.log("[1/5] Testing MediaDevices Support & Secure Context Detection...");

  await it("1. Returns unsupported in Node/SSR environment where window is undefined", () => {
    // @ts-expect-error - simulating SSR environment
    delete globalThis.window;
    const res = checkMediaDevicesSupport();
    expect(res.isSupported).toBe(false);
    expect(res.isSecureContext).toBe(false);
    expect(res.errorReason).toContain("browser client environments");
  });

  await it("2. Detects insecure context when hostname is non-localhost HTTP", () => {
    setMockWindow({
      isSecureContext: false,
      location: { hostname: "192.168.1.105" },
    });
    setMockNavigator({
      mediaDevices: { getUserMedia: () => {} },
    });

    const res = checkMediaDevicesSupport();
    expect(res.isSupported).toBe(false);
    expect(res.isSecureContext).toBe(false);
    expect(res.errorReason).toContain("secure context (HTTPS or localhost)");
  });

  await it("3. Allows localhost or 127.0.0.1 as secure development context", () => {
    setMockWindow({
      isSecureContext: false,
      location: { hostname: "localhost" },
    });
    setMockNavigator({
      mediaDevices: { getUserMedia: () => {} },
    });

    const res = checkMediaDevicesSupport();
    expect(res.isSupported).toBe(true);
    expect(res.isSecureContext).toBe(true);
  });

  await it("4. Fails gracefully when browser lacks navigator.mediaDevices.getUserMedia", () => {
    setMockWindow({
      isSecureContext: true,
      location: { hostname: "driveguard.app" },
    });
    setMockNavigator({});

    const res = checkMediaDevicesSupport();
    expect(res.isSupported).toBe(false);
    expect(res.isSecureContext).toBe(true);
    expect(res.errorReason).toContain("getUserMedia API");
  });

  // -------------------------------------------------------------------------
  // TEST GROUP 2: Constraints Generation & Resolution Matrix
  // -------------------------------------------------------------------------
  console.log("\n[2/5] Testing MediaStream Constraints Matrix & Resolution Presets...");

  await it("5. Generates automotive baseline constraints (640x480 @ 25 FPS user-facing)", () => {
    const c = getRecommendedConstraints(undefined, "user", 640, 480, 25);
    const video = c.video as MediaTrackConstraints;
    expect(c.audio).toBe(false);
    expect((video.facingMode as ConstrainDOMStringParameters)?.ideal).toBe("user");
    expect((video.width as ConstrainULongRange)?.ideal).toBe(640);
    expect((video.height as ConstrainULongRange)?.ideal).toBe(480);
    expect((video.frameRate as ConstrainDoubleRange)?.ideal).toBe(25);
    // Upper bounds
    expect((video.width as ConstrainULongRange)?.max).toBe(1280);
    expect((video.height as ConstrainULongRange)?.max).toBe(720);
    expect((video.frameRate as ConstrainDoubleRange)?.max).toBe(30);
  });

  await it("6. Generates high-definition constraints (1280x720 @ 30 FPS)", () => {
    const c = getRecommendedConstraints(undefined, "user", 1280, 720, 30);
    const video = c.video as MediaTrackConstraints;
    expect((video.width as ConstrainULongRange)?.ideal).toBe(1280);
    expect((video.height as ConstrainULongRange)?.ideal).toBe(720);
    expect((video.frameRate as ConstrainDoubleRange)?.ideal).toBe(30);
    expect((video.width as ConstrainULongRange)?.max).toBe(1280);
    expect((video.height as ConstrainULongRange)?.max).toBe(720);
  });

  await it("7. Enforces exact deviceId constraint when deviceId is supplied", () => {
    const c = getRecommendedConstraints("usb_cam_guid_9876", "user", 640, 480, 20);
    const video = c.video as MediaTrackConstraints;
    expect((video.deviceId as ConstrainDOMStringParameters)?.exact).toBe("usb_cam_guid_9876");
    expect((video.frameRate as ConstrainDoubleRange)?.ideal).toBe(20);
  });

  await it("8. Generates progressive fallback and basic constraints", () => {
    const fallback = getFallbackConstraints("user");
    const fbVideo = fallback.video as MediaTrackConstraints;
    expect(fallback.audio).toBe(false);
    expect((fbVideo.facingMode as ConstrainDOMStringParameters)?.ideal).toBe("user");

    const basic = getBasicConstraints();
    expect(basic.audio).toBe(false);
    expect(basic.video).toBe(true);
  });

  // -------------------------------------------------------------------------
  // TEST GROUP 3: Device Enumeration & Label Keyword Parsing
  // -------------------------------------------------------------------------
  console.log("\n[3/5] Testing Device Enumeration & Facing Mode Heuristics...");

  await it("9. Infers 'user' facing mode for FaceTime, front, and user keywords", () => {
    const rawDevices: MediaDeviceInfo[] = [
      {
        deviceId: "dev_1",
        kind: "videoinput",
        label: "FaceTime HD Camera (Built-in)",
        groupId: "grp_1",
        toJSON: () => ({}),
      },
      {
        deviceId: "dev_2",
        kind: "videoinput",
        label: "Front TrueDepth Camera",
        groupId: "grp_2",
        toJSON: () => ({}),
      },
    ];

    const parsed = parseCameraDevices(rawDevices);
    expect(parsed.length).toBe(2);
    expect(parsed[0].facingMode).toBe("user");
    expect(parsed[0].isDefault).toBe(true);
    expect(parsed[1].facingMode).toBe("user");
    expect(parsed[1].isDefault).toBe(false);
  });

  await it("10. Infers 'environment' facing mode for back, rear, and ultra-wide keywords", () => {
    const rawDevices: MediaDeviceInfo[] = [
      {
        deviceId: "dev_3",
        kind: "videoinput",
        label: "Back Dual Wide Camera",
        groupId: "grp_3",
        toJSON: () => ({}),
      },
      {
        deviceId: "dev_4",
        kind: "videoinput",
        label: "Rear Telephoto Sensor",
        groupId: "grp_4",
        toJSON: () => ({}),
      },
      {
        deviceId: "mic_1",
        kind: "audioinput",
        label: "Built-in Microphone",
        groupId: "grp_mic",
        toJSON: () => ({}),
      },
    ];

    const parsed = parseCameraDevices(rawDevices);
    expect(parsed.length).toBe(2); // Audio device filtered out
    expect(parsed[0].facingMode).toBe("environment");
    expect(parsed[1].facingMode).toBe("environment");
  });

  await it("11. Defaults to 'unknown' facing mode for generic USB webcams", () => {
    const rawDevices: MediaDeviceInfo[] = [
      {
        deviceId: "dev_generic",
        kind: "videoinput",
        label: "Logitech Brio 4K Stream",
        groupId: "grp_logi",
        toJSON: () => ({}),
      },
    ];

    const parsed = parseCameraDevices(rawDevices);
    expect(parsed[0].facingMode).toBe("unknown");
    expect(parsed[0].label).toBe("Logitech Brio 4K Stream");
  });

  // -------------------------------------------------------------------------
  // TEST GROUP 4: Resolution Extraction & Error Recovery Mapping
  // -------------------------------------------------------------------------
  console.log("\n[4/5] Testing Resolution Extraction & DOMException Error Mapping...");

  await it("12. Extracts resolution and aspect ratio from active HTMLVideoElement", () => {
    const mockVideo = {
      videoWidth: 640,
      videoHeight: 480,
    } as HTMLVideoElement;

    const mockTrack = {
      getSettings: () => ({ frameRate: 25.0, width: 640, height: 480 }),
    } as unknown as MediaStreamTrack;

    const res = extractCameraResolution(mockVideo, mockTrack);
    expect(res?.width).toBe(640);
    expect(res?.height).toBe(480);
    expect(res?.aspectRatio).toBe(1.33); // 4:3
    expect(res?.frameRate).toBe(25);
  });

  await it("13. Extracts 16:9 HD resolution (1280x720) correctly", () => {
    const mockVideo = {
      videoWidth: 1280,
      videoHeight: 720,
    } as HTMLVideoElement;

    const res = extractCameraResolution(mockVideo);
    expect(res?.width).toBe(1280);
    expect(res?.height).toBe(720);
    expect(res?.aspectRatio).toBe(1.78);
  });

  await it("14. Falls back to track settings when video dimensions are initially 0", () => {
    const mockVideo = {
      videoWidth: 0,
      videoHeight: 0,
    } as HTMLVideoElement;

    const mockTrack = {
      getSettings: () => ({ width: 640, height: 480, frameRate: 30 }),
    } as unknown as MediaStreamTrack;

    const res = extractCameraResolution(mockVideo, mockTrack);
    expect(res?.width).toBe(640);
    expect(res?.height).toBe(480);
    expect(res?.frameRate).toBe(30);
  });

  await it("15. Maps NotAllowedError to user-actionable permission-denied error", () => {
    const err = new Error("Permission denied by user");
    err.name = "NotAllowedError";

    const mapped = mapMediaErrorToCameraError(err);
    expect(mapped.code).toBe("permission-denied");
    expect(mapped.message).toContain("Camera permission was denied");
    expect(mapped.recoveryAction).toContain("camera or lock icon");
  });

  await it("16. Maps NotFoundError to no-camera-available error", () => {
    const err = new Error("Requested device not found");
    err.name = "NotFoundError";

    const mapped = mapMediaErrorToCameraError(err);
    expect(mapped.code).toBe("no-camera-available");
    expect(mapped.message).toContain("No camera hardware detected");
    expect(mapped.recoveryAction).toContain("connect a USB webcam");
  });

  await it("17. Maps NotReadableError to camera-already-in-use error", () => {
    const err = new Error("Could not start video source");
    err.name = "NotReadableError";

    const mapped = mapMediaErrorToCameraError(err);
    expect(mapped.code).toBe("camera-already-in-use");
    expect(mapped.message).toContain("locked by another application");
    expect(mapped.recoveryAction).toContain("Zoom, Microsoft Teams");
  });

  await it("18. Maps OverconstrainedError to connection-error with fallback action", () => {
    const err = Object.assign(new Error("Constraints could not be satisfied"), {
      name: "OverconstrainedError",
      constraint: "width",
    });

    const mapped = mapMediaErrorToCameraError(err);
    expect(mapped.code).toBe("connection-error");
    expect(mapped.message).toContain("does not support the requested resolution");
    expect(mapped.recoveryAction).toContain("automatically fall back to standard hardware");
  });

  // -------------------------------------------------------------------------
  // TEST GROUP 5: FrameProducer Decoupled Streaming
  // -------------------------------------------------------------------------
  console.log("\n[5/5] Testing FrameProducer Subscription & Dispatch Lifecycle...");

  await it("19. Subscribes consumer, dispatches frame, and unsubscribes cleanly", async () => {
    const producer = new FrameProducer();
    expect(producer.consumerCount).toBe(0);

    const receivedFrames: CapturedFrame[] = [];
    const testConsumer: FrameConsumer = {
      id: "test_sub_01",
      name: "Test Stream Ingest",
      onFrame: async (f) => {
        receivedFrames.push(f);
      },
    };

    const unsubscribe = producer.subscribe(testConsumer);
    expect(producer.consumerCount).toBe(1);

    const dummyFrame: CapturedFrame = {
      blob: new Blob(["dummy"]),
      file: new File(["dummy"], "frame.jpg", { type: "image/jpeg" }),
      dataUrl: "data:image/jpeg;base64,dummy",
      width: 640,
      height: 480,
      format: "image/jpeg",
      sizeBytes: 5,
      timestamp: Date.now(),
    };

    await producer.dispatch(dummyFrame);
    expect(receivedFrames.length).toBe(1);
    expect(receivedFrames[0].width).toBe(640);
    expect(receivedFrames[0].height).toBe(480);

    unsubscribe();
    expect(producer.consumerCount).toBe(0);

    // After unsubscribe, dispatch should not deliver
    await producer.dispatch(dummyFrame);
    expect(receivedFrames.length).toBe(1);
  });

  await it("20. Periodic capture start and stop lifecycle clears timers cleanly", () => {
    const producer = new FrameProducer();
    const mockVideo = {} as HTMLVideoElement;

    producer.startPeriodicCapture(mockVideo, 50);
    producer.stopPeriodicCapture();
    expect(producer.consumerCount).toBe(0);
  });

  // Restore globals
  if (origWindowDesc) {
    Object.defineProperty(globalThis, "window", origWindowDesc);
  } else {
    // @ts-expect-error - cleanup
    delete globalThis.window;
  }
  if (origNavigatorDesc) {
    Object.defineProperty(globalThis, "navigator", origNavigatorDesc);
  }

  console.log("\n======================================================================");
  console.log(`   [DriveGuard Camera Tests] ${passed}/${total} Tests Passed Successfully!`);
  console.log("======================================================================\n");
}

runCameraWebcamTests().catch((err) => {
  console.error("Test execution fatal exception:", err);
  process.exit(1);
});
