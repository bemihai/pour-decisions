/**
 * ChartCard — shared card wrapper for all Recharts-based charts.
 *
 * Provides: Card shell, semantic title, optional one-line description,
 * and a consistent empty-state placeholder.  Used by CellarStatistics
 * and TasteAnalytics.
 */

import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export interface ChartCardProps {
  title: string;
  /** One-line description shown below the title in muted text. */
  description?: string;
  isEmpty?: boolean;
  emptyMessage?: string;
  children: React.ReactNode;
  className?: string;
}

export default function ChartCard({
  title,
  description,
  isEmpty,
  emptyMessage = "No data available.",
  children,
  className,
}: ChartCardProps) {
  return (
    <Card className={cn("flex flex-col", className)}>
      <CardHeader className="pb-1">
        <CardTitle className="text-sm font-semibold leading-snug">{title}</CardTitle>
        {description && (
          <p className="type-caption text-muted-foreground leading-snug">{description}</p>
        )}
      </CardHeader>
      <CardContent className={cn("flex-1 pt-0", isEmpty ? "pb-4" : "p-0 pb-2")}>
        {isEmpty ? (
          <p className="flex items-center justify-center py-10 text-sm text-muted-foreground text-center px-4">
            {emptyMessage}
          </p>
        ) : (
          children
        )}
      </CardContent>
    </Card>
  );
}

