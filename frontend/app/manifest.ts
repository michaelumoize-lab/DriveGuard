import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    id: "/",
    name: "DriveGuard — AI Driver Drowsiness Detection",
    short_name: "DriveGuard",
    description:
      "Personalised Adaptive AI Driver Drowsiness Detection & Real-Time Fatigue Early Warning System.",
    start_url: "/",
    scope: "/",
    display: "standalone",
    orientation: "any",
    background_color: "#101211",
    theme_color: "#101211",
    categories: ["utilities", "productivity", "navigation"],
    icons: [
      {
        src: "/icons/icon-192.png",
        sizes: "192x192",
        type: "image/png",
        purpose: "any",
      },
      {
        src: "/icons/icon-512.png",
        sizes: "512x512",
        type: "image/png",
        purpose: "any",
      },
      {
        src: "/icons/icon-maskable-192.png",
        sizes: "192x192",
        type: "image/png",
        purpose: "maskable",
      },
      {
        src: "/icons/icon-maskable-512.png",
        sizes: "512x512",
        type: "image/png",
        purpose: "maskable",
      },
    ],
    shortcuts: [
      {
        name: "Live Monitor",
        short_name: "Monitor",
        description: "Open the real-time driver drowsiness cockpit",
        url: "/dashboard/live-monitor",
        icons: [
          {
            src: "/icons/icon-192.png",
            sizes: "192x192",
            type: "image/png",
          },
        ],
      },
      {
        name: "Offline Local Monitor",
        short_name: "Local Monitor",
        description: "Run device-isolated offline monitoring without server auth",
        url: "/local-monitor",
        icons: [
          {
            src: "/icons/icon-192.png",
            sizes: "192x192",
            type: "image/png",
          },
        ],
      },
      {
        name: "4-Zone Calibration",
        short_name: "Calibrate",
        description: "Run 4-zone personalized eye and posture calibration",
        url: "/dashboard/calibration",
        icons: [
          {
            src: "/icons/icon-192.png",
            sizes: "192x192",
            type: "image/png",
          },
        ],
      },
    ],
  };
}
