/**
 * Chart configuration — shared constants for all chart components.
 *
 * Centralises dimensions, margins, font settings, and color palettes so that
 * CellarStatistics and TasteAnalytics (and future Recharts migrations) all
 * render consistently. Import from here instead of defining local constants.
 *
 * Color values are sourced from lib/design-tokens.ts, which mirrors the CSS
 * custom properties in app/globals.css.
 */

import {
  BRAND_COLORS,
  CHART_PALETTE,
  RATING_TIER_COLORS,
  WINE_TYPE_COLOR_DEFAULT,
  WINE_TYPE_COLOR_MAP,
} from "./design-tokens";

// ---------------------------------------------------------------------------
// Dimensions
// ---------------------------------------------------------------------------

/** Standard chart heights in pixels. Use CHART_HEIGHT.default for most charts. */
export const CHART_HEIGHT = {
  compact: 240,
  default: 300,
  tall: 360,
} as const;

// ---------------------------------------------------------------------------
// Margins
// ---------------------------------------------------------------------------

/** Plotly margin objects. */
export const CHART_MARGIN = {
  /** Tight margins for charts with no visible axes. */
  tight: { t: 10, b: 10, l: 10, r: 10 },
  /** Comfortable margins when axis labels are shown. */
  withAxes: { t: 10, b: 40, l: 50, r: 10 },
} as const;

/** Recharts margin objects (used in <BarChart margin={...}> etc.). */
export const RECHARTS_MARGIN = {
  tight: { top: 8, right: 8, bottom: 8, left: 8 },
  withAxes: { top: 8, right: 16, bottom: 32, left: 48 },
} as const;

// ---------------------------------------------------------------------------
// Typography
// ---------------------------------------------------------------------------

/** Shared font settings for Plotly layout.font. */
export const CHART_FONT = {
  family: "'Poppins', sans-serif",
  size: 11,
  /** Approximates --muted-foreground in light mode. */
  color: "#8a7f77",
} as const;

// ---------------------------------------------------------------------------
// Recharts responsive container defaults
// ---------------------------------------------------------------------------

export const RECHARTS_CONTAINER = {
  width: "100%",
  /** Default aspect ratio (width / height). Override per chart as needed. */
  aspect: 1.5,
} as const;

// ---------------------------------------------------------------------------
// Plotly base layout
// ---------------------------------------------------------------------------

/**
 * Base Plotly layout object shared across all charts. Spread this first, then
 * override per-chart fields (title, xaxis, yaxis, height, etc.).
 *
 * @example
 * const layout: Partial<PlotlyLayout> = {
 *   ...PLOTLY_LAYOUT_BASE,
 *   height: CHART_HEIGHT.tall,
 *   xaxis: { title: "Vintage" },
 * };
 */
export const PLOTLY_LAYOUT_BASE = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: CHART_FONT,
  showlegend: false,
  margin: CHART_MARGIN.tight,
} as const;

// ---------------------------------------------------------------------------
// Named chart accent colors
// ---------------------------------------------------------------------------

/**
 * Named accent colors for charts that use a single series color.
 * Replaces local PURPLE / GREEN / BROWN constants in CellarStatistics and
 * TasteAnalytics.
 */
export const CHART_ACCENT = {
  primary: `rgba(114, 47, 55, 0.85)`,   // burgundy  (#722F37)
  secondary: `rgba(196, 154, 108, 0.85)`, // gold      (#C49A6C)
  tertiary: `rgba(107, 76, 122, 0.85)`,  // purple    (#6B4C7A)
  green: `rgba(91, 140, 42, 0.85)`,      // green     (#5b8c2a)
  rose: `rgba(194, 114, 122, 0.85)`,     // rosé      (#C2727A)
} as const;

/**
 * Dimmed version of the primary accent (used for secondary series or
 * background fills alongside a highlighted series).
 */
export const CHART_ACCENT_DIM = {
  primary: `rgba(114, 47, 55, 0.3)`,
  secondary: `rgba(196, 154, 108, 0.3)`,
  tertiary: `rgba(107, 76, 122, 0.3)`,
} as const;

// ---------------------------------------------------------------------------
// Helper: wine type → chart rgba color
// ---------------------------------------------------------------------------

/**
 * Returns the rgba color string for a given wine type, suitable for use in
 * Plotly `marker.color` arrays or Recharts `fill` props.
 *
 * @example
 * const colors = typeLabels.map(wineTypeRgba);
 */
export function wineTypeRgba(wineType: string): string {
  return WINE_TYPE_COLOR_MAP[wineType]?.rgba ?? WINE_TYPE_COLOR_DEFAULT.rgba;
}

// ---------------------------------------------------------------------------
// Re-exports from design-tokens for chart consumers
// ---------------------------------------------------------------------------

export { CHART_PALETTE, RATING_TIER_COLORS, BRAND_COLORS };

