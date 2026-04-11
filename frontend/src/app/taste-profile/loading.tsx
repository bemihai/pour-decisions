/**
 * Taste Profile page loading skeleton.
 *
 * Next.js shows this automatically while the Server Component resolves its
 * 11 parallel data fetches. Mirrors the visual structure of page.tsx so there
 * is no layout shift when data arrives.
 */

import { cn } from "@/lib/utils";

function Bone({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-lg bg-muted", className)} />;
}

export default function TasteProfileLoading() {
  return (
    <div className="container mx-auto max-w-7xl px-4 py-6">
      {/* Page header */}
      <div className="mb-6 flex flex-col gap-2">
        <Bone className="h-7 w-48" />
        <Bone className="h-4 w-64" />
      </div>

      {/* Tab bar */}
      <div className="mb-6 flex gap-2 rounded-xl border border-border bg-muted/40 p-1">
        <Bone className="h-9 flex-1" />
        <Bone className="h-9 flex-1" />
        <Bone className="h-9 flex-1" />
      </div>

      {/* Metric strip (5 cards) */}
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
          </div>
        ))}
      </div>

      {/* Chart grid (2 columns, 2 rows) */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="rounded-xl ring-1 ring-foreground/10 bg-card p-4 flex flex-col gap-3">
            <Bone className="h-5 w-36" />
            <Bone className="h-[240px] w-full" />
          </div>
        ))}
      </div>
    </div>
  );
}

