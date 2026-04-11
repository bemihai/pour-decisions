/**
 * Apple touch icon — 180×180, used when the user adds the app to their
 * iOS/macOS home screen. Served automatically by Next.js as apple-touch-icon.
 */

import { ImageResponse } from "next/og";

export const runtime = "edge";
export const size = { width: 180, height: 180 };
export const contentType = "image/png";

export default function AppleIcon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: 180,
          height: 180,
          background: "#722F37",
          borderRadius: 36,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <svg
          width="120"
          height="120"
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
    ),
    { width: 180, height: 180 },
  );
}

