/**
 * DrinkingIndex component.
 *
 * Visualises the drinking readiness of a single wine as a colour-coded status
 * badge. Replaces render_drinking_index_bar() and get_drinking_status() from
 * src/ui/helper/display.py.
 */
import type { DrinkStatus } from "@/lib/types";
import { cn, getDrinkingStatus } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";

interface DrinkingIndexProps {
  /** Window-derived status returned by the backend. */
  drinkStatus: DrinkStatus;
  className?: string;
}

export default function DrinkingIndex({ drinkStatus, className }: DrinkingIndexProps) {
  const status = getDrinkingStatus(drinkStatus);

  return (
    <Badge
      variant="outline"
      className={cn(status.colorClass, "border-current font-semibold", className)}
    >
      {status.label}
    </Badge>
  );
}

