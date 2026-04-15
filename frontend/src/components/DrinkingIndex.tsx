/**
 * DrinkingIndex component.
 *
 * Visualises the drinking readiness of a single wine as a colour-coded status
 * badge. Replaces render_drinking_index_bar() and get_drinking_status() from
 * src/ui/helper/display.py.
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

  return (
    <Badge
      variant="outline"
      className={cn(status.colorClass, "border-current font-semibold", className)}
    >
      {status.label}
    </Badge>
  );
}

