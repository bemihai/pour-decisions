/**
 * EmptyState component.
 *
 * Standardised empty-state display with icon, title, optional description,
 * and optional CTA button. Replaces all ad-hoc empty states across the app.
 *
 * Usage:
 *   <EmptyState
 *     icon={Wine}
 *     title="No wines found"
 *     description="Try adjusting the filters or clearing your search."
 *   />
 *
 *   <EmptyState
 *     icon={Wine}
 *     title="Your cellar is empty"
 *     description="Import your wines from CellarTracker to get started."
 *     action={<Button>Import from CellarTracker</Button>}
 *   />
 */

import * as React from "react";
import { cn } from "@/lib/utils";

export interface EmptyStateProps {
  /** Lucide (or any) icon component rendered above the title. */
  icon?: React.ComponentType<{ className?: string }>;
  /** Short headline. */
  title: string;
  /** Supporting explanation below the title. */
  description?: string;
  /** Optional CTA rendered below the description. */
  action?: React.ReactNode;
  /** Extra classes on the outer container. */
  className?: string;
}

export default function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center gap-3 rounded-xl border border-dashed px-6 py-16 text-center",
        className,
      )}
    >
      {Icon && <Icon className="size-9 text-muted-foreground/40" aria-hidden="true" />}

      <div className="flex flex-col gap-1">
        <p className="type-card-title text-foreground">{title}</p>
        {description && (
          <p className="type-body text-muted-foreground max-w-xs mx-auto">{description}</p>
        )}
      </div>

      {action && <div className="mt-1">{action}</div>}
    </div>
  );
}

