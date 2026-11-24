/**
 * Central media slot manifest for the DriveGuard landing page.
 *
 * All slots ship intentionally blank (`src: null`) so the landing page renders
 * refined dark automotive frames beneath interactive computer-vision overlays.
 *
 * When you download your photos or capture real dashboard screenshots:
 * 1. Place the files inside `frontend/public/images/landing/`
 * 2. Set `src` below to `/images/landing/<filename>`
 */

export type LandingMediaSlotKey =
  | "heroDriver"
  | "nightFatigue"
  | "visionFace"
  | "dashboardOverview"
  | "dashboardLiveMonitor";

export interface LandingMediaAsset {
  src: string | null;
  alt: string;
  placeholderFilename: string;
  recommendedAspect: string;
  slotLabel: string;
}

export const LANDING_MEDIA: Record<LandingMediaSlotKey, LandingMediaAsset> = {
  heroDriver: {
    src: null,
    alt: "Driver seated behind a modern steering wheel monitored by DriveGuard optical vision",
    placeholderFilename: "hero-driver.jpg",
    recommendedAspect: "4:3 or 16:11",
    slotLabel: "DRIVER CABIN OPTICAL FEED",
  },
  nightFatigue: {
    src: null,
    alt: "Cinematic nighttime cabin view illustrating early driver fatigue detection",
    placeholderFilename: "night-fatigue.jpg",
    recommendedAspect: "16:9 or 21:9",
    slotLabel: "NIGHT CABIN ENVIRONMENT",
  },
  visionFace: {
    src: null,
    alt: "Driver facial portrait overlaid with DriveGuard landmark and eye-contour tracking",
    placeholderFilename: "vision-face.jpg",
    recommendedAspect: "4:5",
    slotLabel: "OPTICAL SENSOR PORTRAIT",
  },
  dashboardOverview: {
    src: null,
    alt: "DriveGuard web dashboard showing session history, attention trends, and 4-zone calibration profile",
    placeholderFilename: "dashboard-overview.png",
    recommendedAspect: "16:10",
    slotLabel: "REAL APP SCREENSHOT • /dashboard",
  },
  dashboardLiveMonitor: {
    src: null,
    alt: "DriveGuard Live Monitor cockpit showing real-time facial landmarks and telemetry",
    placeholderFilename: "dashboard-live-monitor.png",
    recommendedAspect: "16:10",
    slotLabel: "REAL APP SCREENSHOT • /dashboard/live-monitor",
  },
};
