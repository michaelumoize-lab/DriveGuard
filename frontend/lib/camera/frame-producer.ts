import { CapturedFrame, FrameCaptureOptions, FrameConsumer } from "@/types/camera";
import { captureVideoFrame } from "./capture-frame";

/**
 * Clean, decoupled FrameProducer interface for future AI backend integration.
 * Allows consumers (e.g., WebSocket streaming clients or REST endpoints) to receive
 * frames without coupling camera hardware lifecycle to AI inference.
 */
export class FrameProducer {
  private consumers: Map<string, FrameConsumer> = new Map();
  private intervalId: NodeJS.Timeout | null = null;
  private isProcessing = false;

  /**
   * Registers a frame consumer (e.g., Python FastAPI pipeline connector).
   */
  public subscribe(consumer: FrameConsumer): () => void {
    this.consumers.set(consumer.id, consumer);
    return () => this.unsubscribe(consumer.id);
  }

  /**
   * Unregisters a frame consumer.
   */
  public unsubscribe(consumerId: string): void {
    this.consumers.delete(consumerId);
  }

  /**
   * Dispatches a captured frame to all registered consumers in parallel.
   */
  public async dispatch(frame: CapturedFrame): Promise<void> {
    if (this.consumers.size === 0) return;

    const promises = Array.from(this.consumers.values()).map(async (consumer) => {
      try {
        await consumer.onFrame(frame);
      } catch (err) {
        console.warn(`FrameConsumer '${consumer.name}' (${consumer.id}) error:`, err);
      }
    });

    await Promise.allSettled(promises);
  }

  /**
   * Starts periodic frame capture from an active HTMLVideoElement.
   */
  public startPeriodicCapture(
    videoElement: HTMLVideoElement,
    intervalMs: number = 100, // Default 10 FPS
    options: FrameCaptureOptions = {}
  ): void {
    this.stopPeriodicCapture();

    this.intervalId = setInterval(async () => {
      if (this.isProcessing || this.consumers.size === 0) return;
      this.isProcessing = true;

      try {
        const frame = await captureVideoFrame(videoElement, options);
        if (frame) {
          await this.dispatch(frame);
        }
      } catch (err) {
        console.error("FrameProducer periodic capture error:", err);
      } finally {
        this.isProcessing = false;
      }
    }, intervalMs);
  }

  /**
   * Stops periodic frame capture and clears active timers.
   */
  public stopPeriodicCapture(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
    this.isProcessing = false;
  }

  /**
   * Returns the count of registered consumers.
   */
  public get consumerCount(): number {
    return this.consumers.size;
  }
}

// Global singleton instance for app-wide decoupled AI streaming
export const globalFrameProducer = new FrameProducer();
