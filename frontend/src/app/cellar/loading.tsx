/**
 * Cellar page loading skeleton.
 *
 * Next.js automatically shows this file while the Server Component fetches
 * cellar stats, filter options, and chart data. Mirrors the layout of
 * page.tsx so there is no layout shift when data arrives.
 */

import { cn } from "@/lib/utils";

function Bone({ className }: { className?: string }) {
  return (
    <div className={cn("animate-pulse rounded-lg bg-muted", className)} />
  );
}

export default function CellarLoading() {
  return (
    <div className="container mx-auto max-w-7xl px-4 py-6">
      {/* Header row -------------------------------------------------------- */}
      <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div className="flex flex-col gap-2">
          <Bone className="h-7 w-40" />
          <Bone className="h-4 w-52" />
        </div>
        <Bone className="h-9 w-44" />
      </div>

      {/* Metrics grid (5 cards) -------------------------------------------- */}
      <div className="mb-6 grid grid-cols-2 gap-4 md:grid-cols-5">
        {Array.from({ length: 5 }).map((_, i) => (
          <div
            key={i}
            className={cn(
              "rounded-xl ring-1 ring-foreground/10 bg-card px-4 py-6 flex flex-col items-center gap-2",
              i === 4 && "col-span-2 md:col-span-1",
            )}
          >
            <Bone className="h-3 w-20" />
            <Bone className="h-8 w-14" />
            <Bone className="h-3 w-24" />
          </div>
        ))}
      </div>

      {/* Tab bar ----------------------------------------------------------- */}
      <div className="mb-4 flex gap-2 rounded-xl border border-border bg-muted/40 p-1">
        <Bone className="h-9 flex-1" />
        <Bone className="h-9 flex-1" />
      </div>

      {/* Filter bar placeholder -------------------------------------------- */}
      <div className="mb-4 flex flex-wrap gap-2">
        {Array.from({ length: 5 }).map((_, i) => (
          <Bone key={i} className="h-9 w-36" />
        ))}
      </div>

      {/* Wine card skeletons ----------------------------------------------- */}
      <div className="flex flex-col gap-3">
        {Array.from({ length: 8 }).map((_, i) => (
          <div
            key={i}
            className="rounded-xl ring-1 ring-foreground/10 bg-card px-4 py-4 flex items-center gap-4"
          >
            <div className="flex-1 flex flex-col gap-2">
              <Bone className="h-4 w-3/5" />
              <Bone className="h-3 w-2/5" />
            </div>
            <Bone className="h-6 w-16" />
            <Bone className="h-6 w-10" />
          </div>
        ))}
      </div>
    </div>
  );
}

