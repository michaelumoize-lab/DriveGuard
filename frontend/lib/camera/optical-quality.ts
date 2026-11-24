/**
 * DriveGuard — Client-Side Optical Frame Quality & Environmental Diagnostics.
 *
 * Implements browser-based parity with backend/app/camera/frame_quality.py.
 * Evaluates in-cabin illumination (mean luminance) and lens focus/motion blur
 * (Laplacian variance) on an offscreen downsampled canvas for < 1ms CPU cost.
 */

import {
  BlurCondition,
  LightingCondition,
  OpticalFrameQuality,
  OpticalQualityStatus,
} from "@/types/camera";

export class OpticalQualityAnalyzer {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;
  private sampleWidth: number = 160;
  private sampleHeight: number = 90;

  constructor(sampleWidth: number = 160, sampleHeight: number = 90) {
    this.sampleWidth = sampleWidth;
    this.sampleHeight = sampleHeight;
  }

  private initCanvas(): boolean {
    if (typeof document === "undefined") return false;
    if (!this.canvas) {
      this.canvas = document.createElement("canvas");
      this.canvas.width = this.sampleWidth;
      this.canvas.height = this.sampleHeight;
      this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    }
    return Boolean(this.ctx);
  }

  /**
   * Analyze optical frame quality from a live HTMLVideoElement.
   */
  public analyze(video: HTMLVideoElement): OpticalFrameQuality | null {
    if (!video || video.readyState < 2 || video.videoWidth === 0) {
      return null;
    }

    if (!this.initCanvas() || !this.ctx || !this.canvas) {
      return null;
    }

    try {
      this.ctx.drawImage(video, 0, 0, this.sampleWidth, this.sampleHeight);
      const imgData = this.ctx.getImageData(0, 0, this.sampleWidth, this.sampleHeight);
      const data = imgData.data;
      const totalPixels = this.sampleWidth * this.sampleHeight;

      let sumLuminance = 0;
      const gray = new Float32Array(totalPixels);

      for (let i = 0; i < totalPixels; i++) {
        const idx = i * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        // Standard Rec. 601 Luma
        const lum = 0.299 * r + 0.587 * g + 0.114 * b;
        gray[i] = lum;
        sumLuminance += lum;
      }

      const meanLuminance = sumLuminance / totalPixels;

      // Compute discrete 2D Laplacian: [0, 1, 0; 1, -4, 1; 0, 1, 0]
      const w = this.sampleWidth;
      const h = this.sampleHeight;
      let lapSum = 0;
      let lapCount = 0;
      const lapValues: number[] = [];

      for (let y = 1; y < h - 1; y += 2) {
        // Step by 2 for sub-sampling speed
        const row = y * w;
        for (let x = 1; x < w - 1; x += 2) {
          const center = gray[row + x];
          const lap =
            gray[row - w + x] +
            gray[row + w + x] +
            gray[row + x - 1] +
            gray[row + x + 1] -
            4 * center;
          lapValues.push(lap);
          lapSum += lap;
          lapCount++;
        }
      }

      const lapMean = lapCount > 0 ? lapSum / lapCount : 0;
      let lapVarSum = 0;
      for (let i = 0; i < lapCount; i++) {
        const diff = lapValues[i] - lapMean;
        lapVarSum += diff * diff;
      }
      const blurVariance = lapCount > 0 ? lapVarSum / lapCount : 0;

      // Classify lighting
      let lightingCondition: LightingCondition = "OPTIMAL";
      const reasons: string[] = [];

      if (meanLuminance < 38.0) {
        lightingCondition = "UNDEREXPOSED";
        reasons.push("CABIN_TOO_DARK");
      } else if (meanLuminance > 215.0) {
        lightingCondition = "OVEREXPOSED";
        reasons.push("SUN_OR_HEADLIGHT_GLARE");
      }

      // Classify focus/blur
      let blurCondition: BlurCondition = "SHARP";
      if (blurVariance < 75.0) {
        blurCondition = "BLURRY";
        reasons.push("LENS_DEFOCUS_OR_MOTION_BLUR");
      }

      // Classify overall status
      let overallQuality: OpticalQualityStatus = "ACCEPTABLE";
      if (lightingCondition === "UNDEREXPOSED" && blurCondition === "BLURRY") {
        overallQuality = "SEVERELY_DEGRADED";
      } else if (lightingCondition !== "OPTIMAL" || blurCondition === "BLURRY") {
        overallQuality = "POOR";
      } else if (blurVariance >= 120.0 && meanLuminance >= 65.0 && meanLuminance <= 185.0) {
        overallQuality = "EXCELLENT";
      }

      if (reasons.length === 0) {
        reasons.push("OPTICAL_CONDITIONS_NOMINAL");
      }

      return {
        meanLuminance: Number(meanLuminance.toFixed(1)),
        blurVarianceScore: Number(blurVariance.toFixed(1)),
        lightingCondition,
        blurCondition,
        overallQuality,
        reasonCodes: reasons,
      };
    } catch {
      return null;
    }
  }

  public dispose(): void {
    this.canvas = null;
    this.ctx = null;
  }
}
