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
  compact?: boolean;
  className?: string;
}

export default function PageHeader({ title, subtitle, compact = false, className }: PageHeaderProps) {
  return (
    <div className={cn(compact ? "mb-4" : "mb-8", className)}>
      <h1
        className={cn(
          "font-bold bg-gradient-to-r from-purple-800 via-purple-600 to-purple-700 bg-clip-text text-transparent",
          compact ? "text-2xl" : "text-4xl",
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

