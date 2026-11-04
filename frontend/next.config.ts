import type { NextConfig } from "next";

const remotePatterns: Array<{ protocol: "http" | "https"; hostname: string; pathname: string }> = [
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
        protocol: (parsed.protocol.replace(":", "") as "http" | "https") || "https",
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
};

export default nextConfig;

