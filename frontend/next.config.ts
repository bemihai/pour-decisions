import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone", // For Docker production builds
};

export default nextConfig;
