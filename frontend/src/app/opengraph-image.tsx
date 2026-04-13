/**
 * Root OpenGraph image — 1200×630.
 * Used as the default link-preview image when sharing the app URL.
 * Page-specific OG images can override this by placing an opengraph-image.tsx
 * in the route segment directory.
 */

import { ImageResponse } from "next/og";

export const runtime = "edge";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";
export const alt = "Pour Decisions — RAG-powered wine assistant and cellar manager";

export default function OgImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: 1200,
          height: 630,
          background: "#FAF8F5",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 24,
          fontFamily: "sans-serif",
        }}
      >
        {/* Background decorative elements */}
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            right: 0,
            height: 8,
            background: "#722F37",
          }}
        />

        {/* Logo mark */}
        <div
          style={{
            width: 96,
            height: 96,
            background: "#722F37",
            borderRadius: 20,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <svg
            width="64"
            height="64"
            viewBox="0 0 32 32"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M8 4 C8 4 6 10 6 14 C6 19.523 10.477 24 16 24 C21.523 24 26 19.523 26 14 C26 10 24 4 24 4 Z"
              fill="white"
              opacity="0.95"
            />
            <rect x="15" y="24" width="2" height="4" rx="1" fill="white" opacity="0.9" />
            <rect x="10" y="27.5" width="12" height="2" rx="1" fill="white" opacity="0.9" />
          </svg>
        </div>

        {/* App name */}
        <div
          style={{
            fontSize: 72,
            fontWeight: 700,
            color: "#722F37",
            letterSpacing: "-0.02em",
            lineHeight: 1,
          }}
        >
          Pour Decisions
        </div>

        {/* Tagline */}
        <div
          style={{
            fontSize: 28,
            fontWeight: 400,
            color: "#8a7f77",
            letterSpacing: "0.01em",
          }}
        >
          RAG-powered wine assistant &amp; cellar manager
        </div>

        {/* Gold accent line */}
        <div
          style={{
            width: 120,
            height: 3,
            background: "#C49A6C",
            borderRadius: 2,
            marginTop: 8,
          }}
        />
      </div>
    ),
    { width: 1200, height: 630 },
  );
}

