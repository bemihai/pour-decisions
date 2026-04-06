/**
 * MetricCard component.
 *
 * Reusable KPI display card replacing st.metric() from Streamlit.
 * Renders a centered label, a prominent value in purple, and an optional delta indicator.
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
  className?: string;
}

function getDeltaColor(delta: string): string {
  const trimmed = delta.trim();
  if (trimmed.startsWith("-")) return "text-red-500";
  if (trimmed.startsWith("+")) return "text-emerald-600";
  return "text-muted-foreground";
}

export default function MetricCard({ label, value, delta, className }: MetricCardProps) {
  return (
    <Card className={cn("text-center", className)}>
      <CardContent className="flex flex-col items-center gap-1 pt-4 pb-4">
        <span className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
          {label}
        </span>
        <span className="text-3xl font-bold text-purple-600 dark:text-purple-400 leading-tight">
          {value}
        </span>
        {delta && (
          <span className={cn("text-sm font-medium", getDeltaColor(delta))}>
            {delta}
          </span>
        )}
      </CardContent>
    </Card>
  );
}

