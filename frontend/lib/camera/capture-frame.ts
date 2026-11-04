import { CapturedFrame, FrameCaptureOptions } from "@/types/camera";

/**
 * Captures an individual frame from an active HTMLVideoElement using an offscreen HTMLCanvasElement.
 * Returns a standardized CapturedFrame object containing a Blob, File, and dataUrl.
 * Decoupled from camera lifecycle and UI components.
 */
export async function captureVideoFrame(
  video: HTMLVideoElement,
  options: FrameCaptureOptions = {}
): Promise<CapturedFrame | null> {
  if (!video) return null;

  // Ensure video dimensions are available
  let vw = video.videoWidth;
  let vh = video.videoHeight;
  if (!vw || !vh) {
    await new Promise<void>((resolve) => setTimeout(resolve, 200));
    vw = video.videoWidth;
    vh = video.videoHeight;
  }
  if (!vw || !vh) {
    console.warn("Unable to capture frame: video dimensions are zero.");
    return null;
  }

  const {
    format = "image/jpeg",
    quality = 0.92,
    targetWidth,
    targetHeight,
    mirror = false,
    cropToAspectRatio,
  } = options;

  // Calculate source cropping rectangle
  let sx = 0;
  let sy = 0;
  let sw = vw;
  let sh = vh;

  if (cropToAspectRatio && cropToAspectRatio > 0) {
    const currentRatio = vw / vh;
    if (currentRatio > cropToAspectRatio) {
      // Source is wider than target ratio -> center-crop width
      sw = Math.round(vh * cropToAspectRatio);
      sx = Math.round((vw - sw) / 2);
    } else if (currentRatio < cropToAspectRatio) {
      // Source is taller than target ratio -> center-crop height
      sh = Math.round(vw / cropToAspectRatio);
      sy = Math.round((vh - sh) / 2);
    }
  }

  // Determine destination canvas dimensions
  const destWidth = targetWidth || sw;
  const destHeight = targetHeight || sh;

  // Create an offscreen canvas
  const canvas = document.createElement("canvas");
  canvas.width = destWidth;
  canvas.height = destHeight;

  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  // Configure high-quality image smoothing
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";

  ctx.save();
  if (mirror) {
    // Horizontally flip the canvas so captured frame matches user-facing mirrored preview
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
  }

  ctx.drawImage(video, sx, sy, sw, sh, 0, 0, destWidth, destHeight);
  ctx.restore();

  const timestamp = Date.now();
  const extension = format === "image/png" ? "png" : format === "image/webp" ? "webp" : "jpg";
  const fileName = `driveguard-frame-${timestamp}.${extension}`;

  const blob = await new Promise<Blob | null>((resolve) => {
    canvas.toBlob((b) => resolve(b), format, quality);
  });

  if (!blob) return null;

  const file = new File([blob], fileName, { type: format, lastModified: timestamp });
  const dataUrl = canvas.toDataURL(format, quality);

  return {
    blob,
    file,
    dataUrl,
    width: destWidth,
    height: destHeight,
    format,
    sizeBytes: blob.size,
    timestamp,
  };
}
