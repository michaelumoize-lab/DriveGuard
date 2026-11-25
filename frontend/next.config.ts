import type { NextConfig } from "next";

const remotePatterns: Array<{
  protocol: "http" | "https";
  hostname: string;
  pathname: string;
}> = [
  {
    protocol: "https",
    hostname: "ik.imagekit.io",
    pathname: "/**",
  },
];

if (process.env.IMAGEKIT_URL_ENDPOINT) {
  try {
    const parsed = new URL(process.env.IMAGEKIT_URL_ENDPOINT);
    if (parsed.hostname && parsed.hostname !== "ik.imagekit.io") {
      remotePatterns.push({
        protocol:
          (parsed.protocol.replace(":", "") as "http" | "https") || "https",
        hostname: parsed.hostname,
        pathname: "/**",
      });
    }
  } catch {
    // Ignore invalid URL format
  }
}

const nextConfig: NextConfig = {
  images: {
    remotePatterns,
  },
  allowedDevOrigins: ["192.168.0.104"],
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          {
            key: "X-Content-Type-Options",
            value: "nosniff",
          },
          {
            key: "X-Frame-Options",
            value: "DENY",
          },
          {
            key: "Referrer-Policy",
            value: "strict-origin-when-cross-origin",
          },
        ],
      },
      {
        source: "/sw.js",
        headers: [
          {
            key: "Content-Type",
            value: "application/javascript; charset=utf-8",
          },
          {
            key: "Cache-Control",
            value: "no-cache, no-store, must-revalidate",
          },
          {
            key: "Content-Security-Policy",
            value: "default-src 'self'; script-src 'self'",
          },
          {
            key: "Service-Worker-Allowed",
            value: "/",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
