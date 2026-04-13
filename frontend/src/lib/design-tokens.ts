/**
 * Design tokens — single source of truth for wine-type colors, brand palette,
 * spacing scale, and wine bottle illustration paths.
 *
 * All component files should import from here instead of defining local color
 * constants. Chart-specific constants live in lib/chart-config.ts, which
 * imports from this module.
 *
 * Mirrors the CSS custom properties defined in app/globals.css :root.
 */

// ---------------------------------------------------------------------------
// Wine type color definitions
// ---------------------------------------------------------------------------

/**
 * All color representations needed for a single wine type:
 * - hex: direct CSS / SVG fill values
 * - rgba: chart library strings (Plotly, Recharts fills)
 * - badge: Tailwind classes for outline badge variant (WineCard)
 * - tailwind: Tailwind classes for filled badge variant (TasteHistory)
 */
export interface WineTypeColors {
  hex: string;
  rgba: string;
  badge: string;
  tailwind: string;
}

/** Per-type color definitions. */
export const WINE_TYPE_COLOR_MAP: Record<string, WineTypeColors> = {
  Red: {
    hex: "#8B1A1A",
    rgba: "rgba(139, 26, 26, 0.85)",
    badge:
      "border-red-300 bg-red-50 text-red-700 dark:border-red-700 dark:bg-red-950 dark:text-red-300",
    tailwind: "bg-red-900 text-white",
  },
  White: {
    hex: "#F4E5A1",
    rgba: "rgba(244, 229, 161, 0.85)",
    badge:
      "border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-300",
    tailwind: "bg-yellow-100 text-yellow-900",
  },
  "Rosé": {
    hex: "#FFB6C1",
    rgba: "rgba(255, 182, 193, 0.85)",
    badge:
      "border-pink-300 bg-pink-50 text-pink-700 dark:border-pink-700 dark:bg-pink-950 dark:text-pink-300",
    tailwind: "bg-pink-200 text-pink-900",
  },
  Rose: {
    hex: "#FFB6C1",
    rgba: "rgba(255, 182, 193, 0.85)",
    badge:
      "border-pink-300 bg-pink-50 text-pink-700 dark:border-pink-700 dark:bg-pink-950 dark:text-pink-300",
    tailwind: "bg-pink-200 text-pink-900",
  },
  Sparkling: {
    hex: "#FFD700",
    rgba: "rgba(255, 215, 0, 0.85)",
    badge:
      "border-blue-300 bg-blue-50 text-blue-700 dark:border-blue-700 dark:bg-blue-950 dark:text-blue-300",
    tailwind: "bg-yellow-300 text-yellow-900",
  },
  Dessert: {
    hex: "#DDA15E",
    rgba: "rgba(221, 161, 94, 0.85)",
    badge:
      "border-orange-300 bg-orange-50 text-orange-700 dark:border-orange-700 dark:bg-orange-950 dark:text-orange-300",
    tailwind: "bg-amber-200 text-amber-900",
  },
  Fortified: {
    hex: "#A0522D",
    rgba: "rgba(160, 82, 45, 0.85)",
    badge:
      "border-amber-400 bg-amber-100 text-amber-800 dark:border-amber-600 dark:bg-amber-900 dark:text-amber-200",
    tailwind: "bg-amber-900 text-white",
  },
};

/** Fallback colors for unknown or null wine types. */
export const WINE_TYPE_COLOR_DEFAULT: WineTypeColors = {
  hex: "#6B4C7A",
  rgba: "rgba(107, 76, 122, 0.85)",
  badge:
    "border-purple-300 bg-purple-50 text-purple-700 dark:border-purple-700 dark:bg-purple-950 dark:text-purple-300",
  tailwind: "bg-muted text-muted-foreground",
};

/** Convenience helper — returns the token set for a given wine type string. */
export function getWineTypeColors(wineType: string | null | undefined): WineTypeColors {
  return WINE_TYPE_COLOR_MAP[wineType ?? ""] ?? WINE_TYPE_COLOR_DEFAULT;
}

// ---------------------------------------------------------------------------
// Brand palette (mirrors globals.css :root --brand-* variables)
// ---------------------------------------------------------------------------

export const BRAND_COLORS = {
  burgundy: "#722F37",
  burgundyLight: "#A05060",
  burgundyDark: "#4D1F25",
  gold: "#C49A6C",
  goldLight: "#E8D5B7",
  purple: "#6B4C7A",
  green: "#5b8c2a",
} as const;

// ---------------------------------------------------------------------------
// Chart color palette (mirrors globals.css --chart-* variables)
// ---------------------------------------------------------------------------

/** Ordered palette for charts. Index 0 is the primary brand color. */
export const CHART_PALETTE: readonly string[] = [
  "#722F37", // burgundy
  "#C49A6C", // gold
  "#6B4C7A", // muted purple
  "#C2727A", // rosé
  "#5b8c2a", // earthy green
  "#A05060", // burgundy-light
  "#E8D5B7", // gold-light
];

/**
 * Rating tier colors — high quality (green) → low quality (orange).
 * Used in rating distribution charts. Matches the Python Streamlit version.
 */
export const RATING_TIER_COLORS: readonly string[] = [
  "rgba(46, 125, 50, 0.85)",
  "rgba(67, 160, 71, 0.85)",
  "rgba(124, 179, 66, 0.85)",
  "rgba(253, 216, 53, 0.85)",
  "rgba(255, 179, 0, 0.85)",
  "rgba(245, 124, 0, 0.85)",
];

// ---------------------------------------------------------------------------
// Wine bottle illustration paths
// ---------------------------------------------------------------------------

/**
 * SVG placeholder illustration paths per wine type.
 * Files live in public/wine-illustrations/ and are served statically.
 * These are created in Phase 4B.7; paths are defined here so components
 * can reference them early without breaking if the files are missing.
 */
export const WINE_BOTTLE_ILLUSTRATIONS: Record<string, string> = {
  Red: "/wine-illustrations/bottle-red.svg",
  White: "/wine-illustrations/bottle-white.svg",
  "Rosé": "/wine-illustrations/bottle-rose.svg",
  Rose: "/wine-illustrations/bottle-rose.svg",
  Sparkling: "/wine-illustrations/bottle-sparkling.svg",
  Dessert: "/wine-illustrations/bottle-dessert.svg",
  Fortified: "/wine-illustrations/bottle-fortified.svg",
  default: "/wine-illustrations/bottle-default.svg",
};

export function getWineBottleIllustration(wineType: string | null | undefined): string {
  return WINE_BOTTLE_ILLUSTRATIONS[wineType ?? ""] ?? WINE_BOTTLE_ILLUSTRATIONS.default;
}

// ---------------------------------------------------------------------------
// Spacing scale (Tailwind class references)
// ---------------------------------------------------------------------------

/**
 * Named spacing constants for consistent gap/padding usage.
 * Use these in className strings rather than bare numeric Tailwind values.
 *
 * xs  → gap-1   (4px)   — tight label stacks
 * sm  → gap-1.5 (6px)   — compact rows
 * md  → gap-2   (8px)   — default inline elements
 * lg  → gap-3   (12px)  — card inner spacing
 * xl  → gap-4   (16px)  — between sections within a card
 * 2xl → gap-6   (24px)  — between cards / major sections
 * 3xl → gap-8   (32px)  — page-level section spacing
 */
export const SPACING = {
  xs: "gap-1",
  sm: "gap-1.5",
  md: "gap-2",
  lg: "gap-3",
  xl: "gap-4",
  "2xl": "gap-6",
  "3xl": "gap-8",
} as const;

