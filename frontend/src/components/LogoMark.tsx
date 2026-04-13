/**
 * LogoMark component — SVG wine glass icon used in the navigation bar,
 * the AI chat avatar, and any other branded context.
 *
 * The shape is a simplified wine glass: wide bowl, tapered stem, flat base.
 * Fill color defaults to the brand burgundy via currentColor inheritance.
 */

import * as React from "react";
import { cn } from "@/lib/utils";

export interface LogoMarkProps {
  /** Side length in pixels. The SVG is always square. */
  size?: number;
  className?: string;
  /** Accessible title for screen readers. */
  title?: string;
}

export default function LogoMark({ size = 28, className, title = "Pour Decisions" }: LogoMarkProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-label={title}
      role="img"
      className={cn("shrink-0", className)}
    >
      {title && <title>{title}</title>}
      {/* Bowl — rounded top, tapers to stem */}
      <path
        d="M8 4 C8 4 6 10 6 14 C6 19.523 10.477 24 16 24 C21.523 24 26 19.523 26 14 C26 10 24 4 24 4 Z"
        fill="currentColor"
        opacity="0.92"
      />
      {/* Bowl highlight — subtle inner shine */}
      <path
        d="M11 7 C10.5 9 10 11.5 10 14 C10 17 11.5 19.5 13.5 21"
        stroke="white"
        strokeWidth="1.2"
        strokeLinecap="round"
        opacity="0.35"
      />
      {/* Stem */}
      <rect x="15" y="24" width="2" height="4" rx="1" fill="currentColor" opacity="0.85" />
      {/* Base */}
      <rect x="10" y="27.5" width="12" height="2" rx="1" fill="currentColor" opacity="0.85" />
    </svg>
  );
}

