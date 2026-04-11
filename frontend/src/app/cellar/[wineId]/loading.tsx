/**
 * Wine detail page loading skeleton.
 *
 * Next.js renders this while the Server Component fetches GET /api/wines/:id.
 * Mirrors the layout of page.tsx to minimise layout shift.
 */

import { cn } from "@/lib/utils";

function Bone({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-lg bg-muted", className)} />;
}

export default function WineDetailLoading() {
  return (
    <div className="container mx-auto max-w-7xl px-4 py-6">
      {/* Breadcrumbs */}
      <Bone className="mb-4 h-4 w-64" />

      {/* Wine header */}
      <div className="mb-8 overflow-hidden rounded-xl border border-border bg-card p-6">
        <div className="flex items-start gap-6">
          <Bone className="hidden sm:block h-28 w-12" />
          <div className="flex-1 flex flex-col gap-3">
            <Bone className="h-3.5 w-36" />
            <Bone className="h-8 w-72" />
            <Bone className="h-3.5 w-48" />
            <div className="flex gap-2">
              <Bone className="h-6 w-16" />
              <Bone className="h-6 w-24" />
              <Bone className="h-6 w-28" />
            </div>
          </div>
          <div className="flex flex-col items-end gap-4 shrink-0">
            <Bone className="h-8 w-16" />
            <Bone className="h-8 w-28" />
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="flex flex-col gap-6 lg:col-span-2">
          <Bone className="h-40 w-full rounded-xl" />
          <Bone className="h-20 w-full rounded-xl" />
          <Bone className="h-48 w-full rounded-xl" />
        </div>
        <div className="flex flex-col gap-6">
          <Bone className="h-36 w-full rounded-xl" />
          <Bone className="h-48 w-full rounded-xl" />
        </div>
      </div>
    </div>
  );
}

