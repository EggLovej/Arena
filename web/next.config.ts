import type { NextConfig } from "next";

// Every page is generated at build time from the JSON in public/data, which the
// tournament workflow rewrites; nothing runs on the server.
const nextConfig: NextConfig = {
  poweredByHeader: false,
};

export default nextConfig;
