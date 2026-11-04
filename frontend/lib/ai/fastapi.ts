import "dotenv/config";
import { z } from "zod";

export interface FastAPILivenessDetail {
  status: "live" | "spoof_detected" | "inconclusive" | "error";
  model: string;
  liveScore: number;
  threshold?: number | null;
  liveThreshold?: number | null;
  spoofThreshold?: number | null;
  individualScores: Record<string, number>;
  processingTimeMs: number;
  ensembleMethod?: string;
}

export interface FastAPIEmbeddingResult {
  success: boolean;
  faceDetected: boolean;
  embedding: number[];
  embeddingSize: number;
  normalized: boolean;
  qualityScore: number;
  detectionConfidence: number;
  model: string;
  liveness?: FastAPILivenessDetail | null;
  executionTimeMs: number;
}

export class FastAPIError extends Error {
  code:
    | "FASTAPI_UNAVAILABLE"
    | "NO_FACE_DETECTED"
    | "MULTIPLE_FACES"
    | "POOR_QUALITY"
    | "INVALID_IMAGE"
    | "AI_SERVICE_ERROR";
  statusCode: number;

  constructor(
    code:
      | "FASTAPI_UNAVAILABLE"
      | "NO_FACE_DETECTED"
      | "MULTIPLE_FACES"
      | "POOR_QUALITY"
      | "INVALID_IMAGE"
      | "AI_SERVICE_ERROR",
    message: string,
    statusCode: number = 400
  ) {
    super(message);
    this.name = "FastAPIError";
    this.code = code;
    this.statusCode = statusCode;
  }
}

// ---------------------------------------------------------------------------
// Strict Zod Runtime Validation Schemas for FastAPI Microservice Responses
// ---------------------------------------------------------------------------

export const faceEmbeddingVectorSchema = z
  .array(z.number().finite())
  .length(512, "Biometric embedding vector must contain exactly 512 dimensions");

export const fastAPILivenessDetailSchema = z.object({
  status: z.enum(["live", "spoof_detected", "inconclusive", "error"]),
  model: z.string(),
  live_score: z.number().finite().min(0.0).max(1.0),
  threshold: z.number().finite().nullable().optional(),
  live_threshold: z.number().finite().nullable().optional(),
  spoof_threshold: z.number().finite().nullable().optional(),
  individual_scores: z.record(z.string(), z.number().finite()).default({}),
  processing_time_ms: z.number().finite().nonnegative().default(0),
  ensemble_method: z.string().optional(),
  error_message: z.string().nullable().optional(),
});

export const fastAPIEmbeddingResponseSchema = z.object({
  success: z.boolean(),
  face_detected: z.boolean(),
  embedding: faceEmbeddingVectorSchema,
  embedding_size: z.number().int().optional().default(512),
  normalized: z.boolean().optional().default(true),
  quality_score: z.number().finite().min(0.0).max(1.0),
  detection_confidence: z.number().finite().min(0.0).max(1.0),
  model: z.string().optional().default("buffalo_s"),
  model_version: z.string().nullable().optional(),
  liveness: fastAPILivenessDetailSchema.nullable().optional(),
  execution_time_ms: z.number().finite().nonnegative().default(0),
});

export const fastAPIDetectedFaceSchema = z.object({
  bbox: z.array(z.number().finite()).default([]),
  confidence: z.number().finite().default(0),
  landmarks: z.array(z.array(z.number().finite())).optional(),
});

export const fastAPIDetectedPersonSchema = z.object({
  bbox: z.array(z.number().finite()).default([]),
  confidence: z.number().finite().default(0),
  is_primary: z.boolean().default(false),
});

export const fastAPIDetectionResponseSchema = z.object({
  success: z.boolean(),
  face_count: z.number().int().nonnegative().default(0),
  faces: z.array(fastAPIDetectedFaceSchema).default([]),
  execution_time_ms: z.number().finite().nonnegative().optional().default(0),
});

export const fastAPIAnalysisResponseSchema = z.object({
  success: z.boolean(),
  status: z.enum([
    "no_person",
    "person_no_face",
    "face_low_quality",
    "spoof_detected",
    "liveness_inconclusive",
    "face_detected_processing",
    "known_individual",
    "unknown_individual",
    "recognition_error",
  ]),
  person_detected: z.boolean(),
  face_detected: z.boolean(),
  person_count: z.number().int().nonnegative().default(0),
  face_count: z.number().int().nonnegative().default(0),
  persons: z.array(fastAPIDetectedPersonSchema).default([]),
  faces: z.array(fastAPIDetectedFaceSchema).default([]),
  primary_person_bbox: z.array(z.number().finite()).length(4).nullable().optional(),
  primary_face_bbox: z.array(z.number().finite()).length(4).nullable().optional(),
  embedding: faceEmbeddingVectorSchema.nullable().optional(),
  embedding_size: z.number().int().nullable().optional(),
  quality_score: z.number().finite().min(0.0).max(1.0).nullable().optional(),
  detection_confidence: z.number().finite().min(0.0).max(1.0).nullable().optional(),
  model: z.string().optional().default("buffalo_s"),
  image_width: z.number().int().positive().default(640),
  image_height: z.number().int().positive().default(480),
  message: z.string().optional(),
  execution_time_ms: z.number().finite().nonnegative().default(0),
  yolo_time_ms: z.number().finite().nonnegative().optional(),
  insightface_time_ms: z.number().finite().nonnegative().optional(),
  antispoof_time_ms: z.number().finite().nonnegative().optional(),
  liveness: fastAPILivenessDetailSchema.nullable().optional(),
  tailgating_detected: z.boolean().default(false),
  tailgating: z
    .object({
      detected: z.boolean(),
      trailing_count: z.number().int().default(0),
      confidence: z.number().finite().default(0),
      risk_level: z.string().default("none"),
      message: z.string().nullable().optional(),
      proximity_ratio: z.number().finite().nullable().optional(),
    })
    .nullable()
    .optional(),
});

function getServiceHeaders(correlationId?: string): Record<string, string> {
  const headers: Record<string, string> = {};
  const apiKey = (process.env.AI_SERVICE_API_KEY || process.env.FASTAPI_API_KEY || "").trim();
  if (apiKey) {
    headers["X-Service-Key"] = apiKey;
  }
  if (correlationId) {
    headers["X-Correlation-ID"] = correlationId;
  }
  return headers;
}

/**
 * Server-side client function to request 512D facial embedding generation from FastAPI AI microservice.
 * Enforces strict runtime Zod response validation and biometric quality standards.
 */
export async function generateEmbedding(
  imageBuffer: Buffer,
  filename: string = "face.jpg",
  correlationId?: string
): Promise<FastAPIEmbeddingResult> {
  const baseUrl = process.env.FASTAPI_URL || "http://localhost:8000";
  const minQualityThreshold = parseFloat(process.env.MIN_FACE_QUALITY || "0.50");

  const endpoint = `${baseUrl.replace(/\/$/, "")}/api/v1/embedding`;

  const formData = new FormData();
  const blob = new Blob([new Uint8Array(imageBuffer)], { type: "image/jpeg" });
  formData.append("file", blob, filename);

  let response: Response;
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout for CPU inference

    response = await fetch(endpoint, {
      method: "POST",
      headers: getServiceHeaders(correlationId),
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
  } catch (err: unknown) {
    console.error("FastAPI service connection error:", err);
    throw new FastAPIError(
      "FASTAPI_UNAVAILABLE",
      "AI face processing service is currently unavailable or unreachable.",
      503
    );
  }

  if (response.status === 503) {
    throw new FastAPIError(
      "FASTAPI_UNAVAILABLE",
      "Face embedding service is starting up or in degraded state.",
      503
    );
  }

  if (!response.ok) {
    let errorDetail = "Failed to process face image.";
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        errorDetail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch (_) {}

    if (errorDetail.includes("No face detected")) {
      throw new FastAPIError("NO_FACE_DETECTED", errorDetail, 400);
    } else if (errorDetail.includes("Multiple faces detected") || errorDetail.includes("one face is required")) {
      throw new FastAPIError("MULTIPLE_FACES", errorDetail, 400);
    } else if (
      errorDetail.includes("enrollment quality") ||
      errorDetail.includes("too dark") ||
      errorDetail.includes("too blurry") ||
      errorDetail.includes("angle is too extreme") ||
      errorDetail.includes("partially cut off") ||
      errorDetail.includes("quality is too low")
    ) {
      throw new FastAPIError("POOR_QUALITY", errorDetail, 400);
    } else if (response.status === 400) {
      throw new FastAPIError("INVALID_IMAGE", errorDetail, 400);
    } else {
      throw new FastAPIError("AI_SERVICE_ERROR", errorDetail, response.status);
    }
  }

  let rawData: unknown;
  try {
    rawData = await response.json();
  } catch {
    throw new FastAPIError(
      "AI_SERVICE_ERROR",
      "AI face processing service returned an unparseable response.",
      502
    );
  }

  const parseResult = fastAPIEmbeddingResponseSchema.safeParse(rawData);
  if (!parseResult.success) {
    const issueSummary = parseResult.error.issues
      .map((i) => `${i.path.join(".") || "root"}: ${i.message}`)
      .join("; ");
    console.error("FastAPI generateEmbedding response validation failure:", issueSummary);
    throw new FastAPIError(
      "AI_SERVICE_ERROR",
      `AI service response validation failed: ${issueSummary}`,
      502
    );
  }

  const data = parseResult.data;

  if (data.quality_score < minQualityThreshold) {
    throw new FastAPIError(
      "POOR_QUALITY",
      `Face image quality score is too low (${data.quality_score.toFixed(2)} < required minimum ${minQualityThreshold.toFixed(2)}). Please upload a clearer, well-lit photo.`,
      400
    );
  }

  return {
    success: true,
    faceDetected: true,
    embedding: data.embedding,
    embeddingSize: data.embedding_size,
    normalized: data.normalized,
    qualityScore: data.quality_score,
    detectionConfidence: data.detection_confidence,
    model: data.model,
    liveness: data.liveness
      ? {
          status: data.liveness.status,
          model: data.liveness.model,
          liveScore: data.liveness.live_score,
          threshold: data.liveness.threshold,
          liveThreshold: data.liveness.live_threshold,
          spoofThreshold: data.liveness.spoof_threshold,
          individualScores: data.liveness.individual_scores,
          processingTimeMs: data.liveness.processing_time_ms,
          ensembleMethod: data.liveness.ensemble_method,
        }
      : null,
    executionTimeMs: data.execution_time_ms,
  };
}

export interface DetectedFaceItem {
  bbox: number[];
  confidence: number;
  landmarks?: number[][];
}

export interface FaceDetectionResult {
  success: boolean;
  faceCount: number;
  faces: DetectedFaceItem[];
  executionTimeMs: number;
}

/**
 * Server-side client function to run face detection and landmark pre-check on an image.
 */
export async function detectFaces(
  imageBuffer: Buffer,
  filename: string = "face.jpg",
  correlationId?: string
): Promise<FaceDetectionResult> {
  const baseUrl = process.env.FASTAPI_URL || "http://localhost:8000";
  const endpoint = `${baseUrl.replace(/\/$/, "")}/api/v1/detect`;

  const formData = new FormData();
  const blob = new Blob([new Uint8Array(imageBuffer)], { type: "image/jpeg" });
  formData.append("file", blob, filename);

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 15000);

    const response = await fetch(endpoint, {
      method: "POST",
      headers: getServiceHeaders(correlationId),
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      const errText = await response.text().catch(() => "");
      throw new Error(`AI Detection responded with status ${response.status}: ${errText}`);
    }

    const rawData = await response.json();
    const parseResult = fastAPIDetectionResponseSchema.safeParse(rawData);
    if (!parseResult.success) {
      throw new Error("AI Detection response schema validation failed.");
    }

    const data = parseResult.data;
    return {
      success: data.success,
      faceCount: data.face_count,
      faces: data.faces.map((f) => ({
        bbox: f.bbox,
        confidence: f.confidence,
        landmarks: f.landmarks,
      })),
      executionTimeMs: data.execution_time_ms,
    };
  } catch (err: unknown) {
    console.error("FastAPI detectFaces error:", err);
    throw err;
  }
}

export interface DetectedPersonItem {
  bbox: number[];
  confidence: number;
  is_primary: boolean;
}

export interface FastAPIAnalysisResult {
  success: boolean;
  status:
    | "no_person"
    | "person_no_face"
    | "face_low_quality"
    | "spoof_detected"
    | "liveness_inconclusive"
    | "face_detected_processing"
    | "known_individual"
    | "unknown_individual"
    | "recognition_error";
  personDetected: boolean;
  faceDetected: boolean;
  personCount: number;
  faceCount: number;
  persons: DetectedPersonItem[];
  faces: DetectedFaceItem[];
  primaryPersonBbox?: number[] | null;
  primaryFaceBbox?: number[] | null;
  embedding?: number[] | null;
  embeddingSize?: number | null;
  qualityScore?: number | null;
  detectionConfidence?: number | null;
  model?: string;
  imageWidth: number;
  imageHeight: number;
  message?: string;
  executionTimeMs: number;
  yoloTimeMs?: number;
  insightfaceTimeMs?: number;
  antispoofTimeMs?: number;
  liveness?: FastAPILivenessDetail | null;
  tailgatingDetected: boolean;
  tailgating?: {
    detected: boolean;
    trailingCount: number;
    confidence: number;
    riskLevel: string;
    message?: string | null;
    proximityRatio?: number | null;
  } | null;
}

/**
 * Server-side client function to request unified YOLO person detection,
 * InsightFace face detection, and embedding extraction from FastAPI.
 * Enforces strict runtime Zod response validation.
 */
export async function analyzeImage(
  imageBuffer: Buffer,
  filename: string = "frame.jpg",
  correlationId?: string
): Promise<FastAPIAnalysisResult> {
  const baseUrl = process.env.FASTAPI_URL || "http://localhost:8000";
  const endpoint = `${baseUrl.replace(/\/$/, "")}/api/v1/analyze`;

  const formData = new FormData();
  const blob = new Blob([new Uint8Array(imageBuffer)], { type: "image/jpeg" });
  formData.append("file", blob, filename);

  let response: Response;
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);

    response = await fetch(endpoint, {
      method: "POST",
      headers: getServiceHeaders(correlationId),
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
  } catch (err: unknown) {
    console.error("FastAPI analyzeImage connection error:", err);
    throw new FastAPIError(
      "FASTAPI_UNAVAILABLE",
      "AI computer vision service is currently unavailable or unreachable.",
      503
    );
  }

  if (response.status === 503) {
    throw new FastAPIError(
      "FASTAPI_UNAVAILABLE",
      "AI computer vision service is starting up or in degraded state.",
      503
    );
  }

  if (!response.ok) {
    let errorDetail = "Failed to analyze image.";
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        errorDetail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {
      // Ignore JSON parse errors for non-JSON response bodies
    }
    if (response.status === 400) {
      throw new FastAPIError("INVALID_IMAGE", errorDetail, 400);
    }
    throw new FastAPIError("AI_SERVICE_ERROR", errorDetail, response.status);
  }

  let rawData: unknown;
  try {
    rawData = await response.json();
  } catch {
    throw new FastAPIError(
      "AI_SERVICE_ERROR",
      "AI computer vision service returned an unparseable response.",
      502
    );
  }

  const parseResult = fastAPIAnalysisResponseSchema.safeParse(rawData);
  if (!parseResult.success) {
    const issueSummary = parseResult.error.issues
      .map((i) => `${i.path.join(".") || "root"}: ${i.message}`)
      .join("; ");
    console.error("FastAPI analyzeImage response validation failure:", issueSummary);
    throw new FastAPIError(
      "AI_SERVICE_ERROR",
      `AI analysis response validation failed: ${issueSummary}`,
      502
    );
  }

  const data = parseResult.data;

  return {
    success: data.success,
    status: data.status,
    personDetected: data.person_detected,
    faceDetected: data.face_detected,
    personCount: data.person_count,
    faceCount: data.face_count,
    persons: data.persons.map((p) => ({
      bbox: p.bbox,
      confidence: p.confidence,
      is_primary: p.is_primary,
    })),
    faces: data.faces.map((f) => ({
      bbox: f.bbox,
      confidence: f.confidence,
      landmarks: f.landmarks,
    })),
    primaryPersonBbox: data.primary_person_bbox || null,
    primaryFaceBbox: data.primary_face_bbox || null,
    embedding: data.embedding || null,
    embeddingSize: data.embedding_size || (data.embedding ? data.embedding.length : null),
    qualityScore: data.quality_score ?? null,
    detectionConfidence: data.detection_confidence ?? null,
    model: data.model || "buffalo_s",
    imageWidth: data.image_width,
    imageHeight: data.image_height,
    message: data.message || undefined,
    executionTimeMs: data.execution_time_ms,
    yoloTimeMs: data.yolo_time_ms || 0,
    insightfaceTimeMs: data.insightface_time_ms || 0,
    antispoofTimeMs: data.antispoof_time_ms || 0,
    liveness: data.liveness
      ? {
          status: data.liveness.status,
          model: data.liveness.model,
          liveScore: data.liveness.live_score,
          threshold: data.liveness.threshold,
          liveThreshold: data.liveness.live_threshold,
          spoofThreshold: data.liveness.spoof_threshold,
          individualScores: data.liveness.individual_scores,
          processingTimeMs: data.liveness.processing_time_ms,
          ensembleMethod: data.liveness.ensemble_method,
        }
      : null,
    tailgatingDetected: Boolean(data.tailgating_detected),
    tailgating: data.tailgating
      ? {
          detected: data.tailgating.detected,
          trailingCount: data.tailgating.trailing_count,
          confidence: data.tailgating.confidence,
          riskLevel: data.tailgating.risk_level,
          message: data.tailgating.message || null,
          proximityRatio: data.tailgating.proximity_ratio ?? null,
        }
      : null,
  };
}

export const fastAPILivenessResponseSchema = z.object({
  success: z.boolean(),
  liveness: fastAPILivenessDetailSchema.nullable().optional(),
  face_detected: z.boolean(),
  face_count: z.number().int().nonnegative().default(1),
  bbox: z.array(z.number().finite()).length(4).nullable().optional(),
  primary_face_bbox: z.array(z.number().finite()).length(4).nullable().optional(),
  execution_time_ms: z.number().finite().nonnegative().default(0),
});

export interface FastAPILivenessCheckResult {
  success: boolean;
  liveness: FastAPILivenessDetail | null;
  faceDetected: boolean;
  faceCount: number;
  primaryFaceBbox?: number[] | null;
  executionTimeMs: number;
}

/**
 * Server-side client function to run dedicated Anti-Spoofing / Liveness verification on an image.
 */
export async function checkLiveness(
  imageBuffer: Buffer,
  filename: string = "frame.jpg",
  correlationId?: string
): Promise<FastAPILivenessCheckResult> {
  const baseUrl = process.env.FASTAPI_URL || "http://localhost:8000";
  const endpoint = `${baseUrl.replace(/\/$/, "")}/api/v1/liveness`;

  const formData = new FormData();
  const blob = new Blob([new Uint8Array(imageBuffer)], { type: "image/jpeg" });
  formData.append("file", blob, filename);

  let response: Response;
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 20000);

    response = await fetch(endpoint, {
      method: "POST",
      headers: getServiceHeaders(correlationId),
      body: formData,
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
  } catch (err: unknown) {
    console.error("FastAPI checkLiveness connection error:", err);
    throw new FastAPIError(
      "FASTAPI_UNAVAILABLE",
      "AI anti-spoofing service is currently unavailable or unreachable.",
      503
    );
  }

  if (!response.ok) {
    let errorDetail = "Failed to evaluate liveness.";
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        errorDetail = typeof errJson.detail === "string" ? errJson.detail : JSON.stringify(errJson.detail);
      }
    } catch {
      // Ignore
    }
    throw new FastAPIError("AI_SERVICE_ERROR", errorDetail, response.status);
  }

  let rawData: unknown;
  try {
    rawData = await response.json();
  } catch (parseErr: unknown) {
    throw new FastAPIError(
      "AI_SERVICE_ERROR",
      `Failed to parse AI service response as JSON: ${parseErr instanceof Error ? parseErr.message : String(parseErr)}`,
      502
    );
  }

  const parseResult = fastAPILivenessResponseSchema.safeParse(rawData);
  if (!parseResult.success) {
    const issueSummary = parseResult.error.issues
      .map((i) => `${i.path.join(".") || "root"}: ${i.message}`)
      .join("; ");
    throw new FastAPIError(
      "AI_SERVICE_ERROR",
      `AI liveness response validation failed: ${issueSummary}`,
      502
    );
  }

  const data = parseResult.data;
  return {
    success: data.success,
    liveness: data.liveness
      ? {
          status: data.liveness.status,
          model: data.liveness.model,
          liveScore: data.liveness.live_score,
          threshold: data.liveness.threshold,
          liveThreshold: data.liveness.live_threshold,
          spoofThreshold: data.liveness.spoof_threshold,
          individualScores: data.liveness.individual_scores,
          processingTimeMs: data.liveness.processing_time_ms,
          ensembleMethod: data.liveness.ensemble_method,
        }
      : null,
    faceDetected: data.face_detected,
    faceCount: data.face_count,
    primaryFaceBbox: data.primary_face_bbox ?? data.bbox ?? null,
    executionTimeMs: data.execution_time_ms,
  };
}
