/**
 * MetricCard component.
 *
 * Reusable KPI display card replacing st.metric() from Streamlit.
 * Renders a centered label, a prominent value in brand-burgundy, and an
 * optional delta indicator.
 */
import { cn } from "@/lib/utils";
import { Card, CardContent } from "@/components/ui/card";

interface MetricCardProps {
  /** Descriptive label shown above the value. */
  label: string;
  /** Primary metric value to display. */
  value: string | number;
  /** Optional delta string (e.g. "+3 this month"). Shown in green when positive, red when negative. */
  delta?: string;
  /** Compact left-aligned treatment for a segmented summary band. */
  variant?: "default" | "summary";
  className?: string;
}

function getDeltaColor(delta: string): string {
  const trimmed = delta.trim();
  if (trimmed.startsWith("-")) return "text-red-500";
  if (trimmed.startsWith("+")) return "text-emerald-600";
  return "text-muted-foreground";
}

export default function MetricCard({
  label,
  value,
  delta,
  variant = "default",
  className,
}: MetricCardProps) {
  const isSummary = variant === "summary";

  return (
    <Card
      className={cn(
        isSummary
          ? "min-h-24 rounded-none bg-transparent py-0 text-left shadow-none ring-0"
          : "text-center",
        className,
      )}
    >
      <CardContent
        className={cn(
          "flex flex-col gap-1",
          isSummary ? "items-start justify-center px-5 py-4" : "items-center py-5",
        )}
      >
        <span
          className={cn(
            "text-muted-foreground uppercase tracking-wide",
            isSummary ? "type-caption font-semibold" : "type-label",
          )}
        >
          {label}
        </span>
        <span
          className={cn(
            "font-bold text-brand-burgundy dark:text-brand-burgundy-light leading-tight",
            isSummary ? "text-2xl" : "text-3xl",
          )}
        >
          {value}
        </span>
        {delta && (
          <span className={cn("type-caption font-medium", getDeltaColor(delta))}>
            {delta}
          </span>
        )}
      </CardContent>
    </Card>
  );
}

