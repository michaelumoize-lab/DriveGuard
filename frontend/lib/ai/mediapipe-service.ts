/**
 * DriveGuard — MediaPipe FaceLandmarker Browser Service.
 *
 * Manages the lifecycle of `@mediapipe/tasks-vision` FaceLandmarker:
 * - Singleton initialization ensuring model and WASM are loaded once and reused
 * - Local asset loading (/models/face_landmarker.task and /wasm) with CDN fallback
 * - Hardware acceleration (GPU delegate with automatic fallback to CPU)
 * - Safe client-only execution (never runs in SSR)
 * - Proper resource disposal and error diagnostics
 */

import type { FaceLandmarker, FaceLandmarkerResult } from "@mediapipe/tasks-vision";

export interface MediaPipeConfig {
  modelAssetPath?: string;
  wasmBasePath?: string;
  cdnWasmBasePath?: string;
  delegate?: "GPU" | "CPU";
  numFaces?: number;
  minDetectionConfidence?: number;
  minPresenceConfidence?: number;
  minTrackingConfidence?: number;
}

const DEFAULT_CONFIG: MediaPipeConfig = {
  modelAssetPath: "/models/face_landmarker.task",
  wasmBasePath: "/wasm",
  cdnWasmBasePath: "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm",
  delegate: "GPU",
  numFaces: 1,
  minDetectionConfidence: 0.5,
  minPresenceConfidence: 0.5,
  minTrackingConfidence: 0.5,
};

export class MediaPipeService {
  private static _instance: MediaPipeService | null = null;
  private _landmarker: FaceLandmarker | null = null;
  private _isInitializing: boolean = false;
  private _initPromise: Promise<FaceLandmarker> | null = null;
  private _activeDelegate: "GPU" | "CPU" = "GPU";

  private constructor() {}

  public static getInstance(): MediaPipeService {
    if (!MediaPipeService._instance) {
      MediaPipeService._instance = new MediaPipeService();
    }
    return MediaPipeService._instance;
  }

  public get isReady(): boolean {
    return this._landmarker !== null;
  }

  public get activeDelegate(): "GPU" | "CPU" {
    return this._activeDelegate;
  }

  public async initialize(customConfig?: Partial<MediaPipeConfig>): Promise<FaceLandmarker> {
    if (typeof window === "undefined") {
      throw new Error("MediaPipeService can only be initialized in browser client environment.");
    }

    if (this._landmarker) {
      return this._landmarker;
    }

    if (this._initPromise) {
      return this._initPromise;
    }

    const config = { ...DEFAULT_CONFIG, ...customConfig };

    this._initPromise = (async () => {
      this._isInitializing = true;
      try {
        const { FilesetResolver, FaceLandmarker } = await import("@mediapipe/tasks-vision");

        // 1. Resolve WASM assets: try local /wasm first, fallback to CDN
        let visionFileset;
        try {
          visionFileset = await FilesetResolver.forVisionTasks(config.wasmBasePath!);
        } catch (localWasmErr) {
          console.warn("[MediaPipe] Local WASM load failed, falling back to CDN:", localWasmErr);
          visionFileset = await FilesetResolver.forVisionTasks(config.cdnWasmBasePath!);
        }

        // 2. Try initializing with GPU delegate
        let landmarker: FaceLandmarker | null = null;
        let selectedDelegate: "GPU" | "CPU" = config.delegate || "GPU";

        if (selectedDelegate === "GPU") {
          try {
            landmarker = await FaceLandmarker.createFromOptions(visionFileset, {
              baseOptions: {
                modelAssetPath: config.modelAssetPath!,
                delegate: "GPU",
              },
              runningMode: "VIDEO",
              numFaces: config.numFaces ?? 1,
              minFaceDetectionConfidence: config.minDetectionConfidence ?? 0.5,
              minFacePresenceConfidence: config.minPresenceConfidence ?? 0.5,
              minTrackingConfidence: config.minTrackingConfidence ?? 0.5,
              outputFaceBlendshapes: false,
              outputFacialTransformationMatrixes: true,
            });
            this._activeDelegate = "GPU";
          } catch (gpuErr) {
            console.warn("[MediaPipe] GPU delegate initialization failed, falling back to CPU:", gpuErr);
            selectedDelegate = "CPU";
          }
        }

        // 3. Fallback to CPU delegate if needed
        if (!landmarker) {
          landmarker = await FaceLandmarker.createFromOptions(visionFileset, {
            baseOptions: {
              modelAssetPath: config.modelAssetPath!,
              delegate: "CPU",
            },
            runningMode: "VIDEO",
            numFaces: config.numFaces ?? 1,
            minFaceDetectionConfidence: config.minDetectionConfidence ?? 0.5,
            minFacePresenceConfidence: config.minPresenceConfidence ?? 0.5,
            minTrackingConfidence: config.minTrackingConfidence ?? 0.5,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: true,
          });
          this._activeDelegate = "CPU";
        }

        this._landmarker = landmarker;
        return landmarker;
      } catch (err) {
        console.error("[MediaPipe] Fatal initialization error:", err);
        throw err;
      } finally {
        this._isInitializing = false;
        this._initPromise = null;
      }
    })();

    return this._initPromise;
  }

  public detectForVideo(videoElement: HTMLVideoElement, timestampMs: number): FaceLandmarkerResult | null {
    if (!this._landmarker) {
      return null;
    }
    if (
      !videoElement ||
      videoElement.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      videoElement.videoWidth === 0 ||
      videoElement.videoHeight === 0
    ) {
      return null;
    }

    try {
      return this._landmarker.detectForVideo(videoElement, timestampMs);
    } catch (err) {
      console.warn("[MediaPipe] detectForVideo frame skipped:", err);
      return null;
    }
  }

  public close(): void {
    if (this._landmarker) {
      try {
        this._landmarker.close();
      } catch (err) {
        console.warn("[MediaPipe] Error closing landmarker instance:", err);
      }
      this._landmarker = null;
    }
    this._initPromise = null;
    this._isInitializing = false;
  }
}
