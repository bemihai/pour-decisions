/**
 * DrinkingIndex component.
 *
 * Visualises the drinking readiness of a single wine as a colour-coded status
 * badge and a progress bar.  Replaces render_drinking_index_bar() and
 * get_drinking_status() from src/ui/helper/display.py.
 */
import { cn, getDrinkingStatus } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";

interface DrinkingIndexProps {
  /** Drinking index for this wine (0-100 scale). */
  drinkIndex: number | null | undefined;
  /** All drinking indices in the collection, used for p5/p95 normalisation. */
  allIndices: number[];
  className?: string;
}

export default function DrinkingIndex({ drinkIndex, allIndices, className }: DrinkingIndexProps) {
  if (drinkIndex == null) return null;

  const status = getDrinkingStatus(drinkIndex, allIndices);
  const barLabel = status.normalised >= 50 ? "Drink Sooner" : "Drink Later";

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground">Drinking Readiness</span>
        <Badge
          variant="outline"
          className={cn(status.colorClass, "border-current font-semibold")}
        >
          {status.label}
        </Badge>
      </div>

      {/* Progress bar */}
      <div className="relative h-5 w-full overflow-hidden rounded-full bg-muted">
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-300"
          style={{ width: `${status.normalised}%`, backgroundColor: status.hex }}
        />
        <span className="absolute inset-0 flex items-center justify-center text-xs font-bold text-foreground">
          {barLabel}
        </span>
      </div>
    </div>
  );
}

