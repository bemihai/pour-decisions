"use client";

/**
 * PlotlyChart component.
 *
 * Reusable wrapper around Plotly.js for all charts in the cellar and taste-
 * profile pages.  Dynamically imports plotly.js-dist-min to keep it out of
 * the initial bundle and avoid SSR errors.
 *
 * Usage:
 *   <PlotlyChart data={[{ type: "pie", values: [30, 70], labels: ["Red", "White"] }]} />
 */

import { useEffect, useRef } from "react";
import type * as PlotlyType from "plotly.js";

import { cn } from "@/lib/utils";

// plotly.js carries the TypeScript type definitions; plotly.js-dist-min is the
// slim runtime build (no separate @types).  We import the types from the former
// and load the latter at runtime via a dynamic import below.
type PlotlyRuntime = typeof import("plotly.js");

export interface PlotlyChartProps {
  data: PlotlyType.Data[];
  layout?: Partial<PlotlyType.Layout>;
  className?: string;
}

// ---------------------------------------------------------------------------
// Defaults that match the project's Poppins/transparent design language.
// ---------------------------------------------------------------------------
const DEFAULT_LAYOUT: Partial<PlotlyType.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { family: "Poppins, sans-serif", size: 12 },
  margin: { t: 32, r: 16, b: 40, l: 48 },
  autosize: true,
};

const PLOT_CONFIG: Partial<PlotlyType.Config> = {
  displayModeBar: false,
  responsive: true,
};

// ---------------------------------------------------------------------------
// Module-level cache: load plotly.js-dist-min once per page lifetime.
// ---------------------------------------------------------------------------
let _plotlyPromise: Promise<PlotlyRuntime> | null = null;

function getPlotly(): Promise<PlotlyRuntime> {
  if (!_plotlyPromise) {
    _plotlyPromise = import("plotly.js-dist-min").then((mod) => {
      // plotly.js-dist-min is CJS; webpack exposes the namespace as .default.
      return (mod.default ?? mod) as unknown as PlotlyRuntime;
    });
  }
  return _plotlyPromise;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function PlotlyChart({ data, layout, className }: PlotlyChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    let alive = true;
    const mergedLayout: Partial<PlotlyType.Layout> = { ...DEFAULT_LAYOUT, ...layout };

    getPlotly().then((Plotly) => {
      if (!alive || !el.isConnected) return;
      // Plotly.react() creates the chart on first call and diffs on subsequent
      // calls, which is more efficient than newPlot() + purge() on every render.
      Plotly.react(el, data, mergedLayout, PLOT_CONFIG);
    });

    return () => {
      alive = false;
      // Purge releases WebGL contexts and event listeners.
      getPlotly().then((Plotly) => Plotly.purge(el));
    };
  }, [data, layout]);

  return <div ref={containerRef} className={cn("w-full", className)} />;
}

