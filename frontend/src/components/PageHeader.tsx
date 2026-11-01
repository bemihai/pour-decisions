/**
 * PageHeader component.
 *
 * Reusable page title block with an optional subtitle and a compact variant.
 * Replaces make_page_title() / make_compact_page_title() from src/ui/helper/display.py.
 */
import { cn } from "@/lib/utils";

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  eyebrow?: string;
  compact?: boolean;
  className?: string;
}

export default function PageHeader({
  title,
  subtitle,
  eyebrow,
  compact = false,
  className,
}: PageHeaderProps) {
  return (
    <div className={cn(compact ? "mb-4" : "mb-8", className)}>
      {eyebrow && (
        <p className="mb-1 text-xs font-semibold uppercase tracking-[0.14em] text-brand-burgundy dark:text-brand-burgundy-light">
          {eyebrow}
        </p>
      )}
      <h1
        className={cn(
          "font-heading font-bold tracking-tight text-brand-burgundy dark:text-brand-burgundy-light",
          compact ? "text-2xl" : "type-page-title",
        )}
      >
        {title}
      </h1>
      {subtitle && (
        <p
          className={cn(
            "text-muted-foreground",
            compact ? "mt-1 text-sm" : "mt-2 text-base",
          )}
        >
          {subtitle}
        </p>
      )}
    </div>
  );
}

