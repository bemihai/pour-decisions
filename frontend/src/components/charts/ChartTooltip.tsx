/**
 * ChartTooltip — shared custom tooltip for all Recharts charts.
 *
 * Styled to match the app's card / border design language instead of
 * Recharts' default white-on-gray tooltip.
 */

// Recharts passes these as props to the content component.
export interface ChartTooltipProps {
  active?: boolean;
  payload?: Array<{
    name: string;
    value: number | string;
    fill?: string;
    stroke?: string;
    color?: string;
  }>;
  label?: string | number;
  /** Format the value before display (e.g. append "/100" for ratings). */
  formatter?: (value: number | string, name: string) => string;
  /** When true, the series name is hidden (useful for single-series charts). */
  hideName?: boolean;
}

export default function ChartTooltip({
  active,
  payload,
  label,
  formatter,
  hideName,
}: ChartTooltipProps) {
  if (!active || !payload?.length) return null;

  return (
    <div className="rounded-md border border-border bg-background px-3 py-2 shadow-lg text-xs">
      {label != null && (
        <p className="font-medium text-foreground mb-1">{label}</p>
      )}
      {payload.map((p, i) => {
        const color = p.fill ?? p.stroke ?? p.color ?? "#888";
        const rawValue = p.value;
        const displayValue = formatter ? formatter(rawValue, p.name) : rawValue;
        return (
          <div key={i} className="flex items-center gap-1.5">
            <span
              className="inline-block h-2 w-2 rounded-full shrink-0"
              style={{ backgroundColor: color }}
            />
            {!hideName && p.name && p.name !== "value" && (
              <span className="text-muted-foreground">{p.name}:</span>
            )}
            <span className="font-medium text-foreground">{String(displayValue)}</span>
          </div>
        );
      })}
    </div>
  );
}

