import { Check, CircleEllipsis, TriangleAlert } from "lucide-react";

import type { ToolProgressStreamEvent } from "@/lib/types";
import { cn } from "@/lib/utils";


const TOOL_LABELS: Record<string, string> = {
  get_cellar_wines: "Checking your cellar",
  get_wine_details: "Checking wine details",
  get_cellar_statistics: "Checking cellar statistics",
  get_user_taste_profile: "Checking your taste profile",
  get_top_rated_wines: "Checking top-rated wines",
  get_wine_recommendations_from_profile: "Finding profile matches",
  compare_wine_to_profile: "Comparing with your taste profile",
  get_food_pairing_wines: "Finding food pairings",
  get_pairing_for_wine: "Checking wine pairings",
  get_wine_and_cheese_pairings: "Finding cheese pairings",
  search_wine_knowledge: "Searching wine knowledge",
  search_wine_region_info: "Searching region information",
  search_grape_variety_info: "Searching grape information",
  search_wine_term_definition: "Checking wine terminology",
  search_wine_producer_info: "Searching producer information",
  search_web_for_wine: "Searching current wine information",
  search_wine_price: "Checking current wine prices",
  search_wine_reviews: "Checking wine reviews",
  other: "Using a wine tool",
};

const STATUS_LABELS = {
  started: "In progress",
  completed: "Completed",
  failed: "Could not complete",
} as const;

interface AgentExecutionTimelineProps {
  events: ToolProgressStreamEvent[];
}

export default function AgentExecutionTimeline({ events }: AgentExecutionTimelineProps) {
  if (events.length === 0) return null;

  return (
    <div
      role="status"
      aria-live="polite"
      aria-label="Agent progress"
      className="mb-4 ml-12 max-w-lg rounded-xl border border-border bg-muted/30 px-4 py-3"
    >
      <p className="mb-2 text-xs font-medium text-muted-foreground">Working on your request</p>
      <ul className="space-y-2">
        {events.map((event) => {
          const Icon =
            event.status === "completed"
              ? Check
              : event.status === "failed"
                ? TriangleAlert
                : CircleEllipsis;
          return (
            <li key={event.invocation_id} className="flex items-center gap-2 text-sm">
              <Icon
                aria-hidden
                className={cn(
                  "h-4 w-4 shrink-0",
                  event.status === "completed" && "text-emerald-600",
                  event.status === "failed" && "text-destructive",
                  event.status === "started" && "text-brand-burgundy",
                )}
              />
              <span className="text-foreground">
                {TOOL_LABELS[event.tool_key] ?? TOOL_LABELS.other}
              </span>
              <span className="ml-auto text-xs text-muted-foreground">
                {STATUS_LABELS[event.status]}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
