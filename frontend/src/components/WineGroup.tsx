"use client";

/**
 * WineGroup component — Phase 4C.5.
 *
 * Groups InventoryItems that share the same wine identity (same producer,
 * wine name, and region) but differ only by vintage. The primary card is
 * shown full-size; additional vintages appear as selectable chips below it.
 *
 * Grouping key: `producer_name + wine_name + region_name`
 * Per the design decision, different regions/appellations from the same
 * producer stay as separate groups (e.g., Louis Jadot Chablis vs
 * Louis Jadot Chablis Premier Cru).
 *
 * Usage (from CellarInventory):
 *   const groups = groupWinesByIdentity(items);
 *   groups.map(group => <WineGroup key={group.key} wines={group.wines} allDrinkIndices={...} />)
 */

import { useState } from "react";
import { cn } from "@/lib/utils";
import type { InventoryItem } from "@/lib/types";
import WineCard from "@/components/WineCard";

// ---------------------------------------------------------------------------
// Grouping logic (pure function, exported for use in CellarInventory)
// ---------------------------------------------------------------------------

export interface WineGroupData {
  /** Stable string key for React list rendering. */
  key: string;
  /** All items in the group, sorted by vintage descending (most recent first). */
  wines: InventoryItem[];
}

/** Derive the identity key for a single InventoryItem. */
export function wineIdentityKey(item: InventoryItem): string {
  return [
    item.producer_name ?? "",
    item.wine_name,
    item.region_name ?? "",
  ]
    .join("|")
    .toLowerCase();
}

/**
 * Group a flat list of InventoryItems by wine identity.
 * Returns groups in the order they were first encountered (preserving the
 * server-side sort order of the representative wine).
 */
export function groupWinesByIdentity(items: InventoryItem[]): WineGroupData[] {
  const map = new Map<string, InventoryItem[]>();

  for (const item of items) {
    const key = wineIdentityKey(item);
    const existing = map.get(key);
    if (existing) {
      existing.push(item);
    } else {
      map.set(key, [item]);
    }
  }

  return Array.from(map.entries()).map(([key, wines]) => ({
    key,
    // Sort vintages: most recent first, NV (null) at the end
    wines: wines.sort((a, b) => {
      if (a.vintage == null && b.vintage == null) return 0;
      if (a.vintage == null) return 1;
      if (b.vintage == null) return -1;
      return b.vintage - a.vintage;
    }),
  }));
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface WineGroupProps {
  wines: InventoryItem[];
  allDrinkIndices: number[];
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function WineGroup({ wines, allDrinkIndices }: WineGroupProps) {
  const [activeIndex, setActiveIndex] = useState(0);

  // Guard: single wine — no grouping UI needed
  if (wines.length === 1) {
    return <WineCard wine={wines[0]} allDrinkIndices={allDrinkIndices} />;
  }

  const primary = wines[activeIndex] ?? wines[0];

  return (
    <div className="flex flex-col gap-1">
      <WineCard wine={primary} allDrinkIndices={allDrinkIndices} />

      {/* Vintage selector chips */}
      <div
        className="flex flex-wrap items-center gap-1.5 rounded-b-lg border border-t-0 border-border bg-muted/20 px-4 py-2"
        role="group"
        aria-label="Available vintages"
      >
        <span className="type-caption text-muted-foreground shrink-0">Vintages:</span>
        {wines.map((wine, i) => (
          <button
            key={wine.wine_id}
            onClick={() => setActiveIndex(i)}
            aria-pressed={i === activeIndex}
            className={cn(
              "rounded border px-2 py-0.5 type-caption font-medium transition-colors",
              i === activeIndex
                ? "border-brand-burgundy bg-brand-burgundy text-white"
                : "border-border text-muted-foreground hover:border-brand-burgundy hover:text-brand-burgundy",
            )}
          >
            {wine.vintage ?? "NV"}
            <span className="ml-1 opacity-70">({wine.quantity})</span>
          </button>
        ))}
      </div>
    </div>
  );
}

