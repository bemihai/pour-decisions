/**
 * Section compound component.
 *
 * Wraps a titled content region with consistent spacing and visual treatment.
 * Use this wherever a page contains multiple logical sections (e.g., the
 * metrics strip above the cellar tabs, or the filters + inventory split).
 *
 * Usage:
 *   <Section title="Cellar Overview" description="Your current stock at a glance.">
 *     <MetricCardGrid ... />
 *   </Section>
 *
 *   <Section>
 *     <FilterPanel ... />
 *   </Section>
 */

import * as React from "react";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SectionProps {
  /** Section heading rendered as an <h2>. Omit for anonymous content sections. */
  title?: React.ReactNode;
  /** Short supporting text rendered below the title. */
  description?: string;
  /** Optional node placed on the right side of the header row (e.g., an action button). */
  action?: React.ReactNode;
  /** Section body content. */
  children: React.ReactNode;
  /** Additional classes applied to the outer wrapper. */
  className?: string;
  /** Additional classes applied to the content area below the header. */
  contentClassName?: string;
}

// ---------------------------------------------------------------------------
// Section
// ---------------------------------------------------------------------------

export default function Section({
  title,
  description,
  action,
  children,
  className,
  contentClassName,
}: SectionProps) {
  const hasHeader = title || action;

  return (
    <section className={cn("flex flex-col gap-4", className)}>
      {hasHeader && (
        <div className="flex items-start justify-between gap-4">
          {title && (
            <div className="flex flex-col gap-0.5">
              <h2 className="type-section-title text-foreground">{title}</h2>
              {description && (
                <p className="type-caption text-muted-foreground">{description}</p>
              )}
            </div>
          )}
          {action && <div className="shrink-0">{action}</div>}
        </div>
      )}
      <div className={cn("flex flex-col gap-4", contentClassName)}>{children}</div>
    </section>
  );
}

