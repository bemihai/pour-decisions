"use client";

import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle2, Loader2, XCircle } from "lucide-react";

import { getMergeSuggestions, mergeSuggestion } from "@/lib/api";
import type { MergeEntityType, MergeSuggestion } from "@/lib/types";
import { Button } from "@/components/ui/button";
import EmptyState from "@/components/EmptyState";

interface PendingAction {
  entityType: MergeEntityType;
  keepId: number;
  removeId: number;
  approve: boolean;
}

export default function CellarMergeRecords() {
  const queryClient = useQueryClient();
  const [message, setMessage] = useState<string | null>(null);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());

  const { data, isLoading, isError, error, refetch } = useQuery({
    queryKey: ["merge-suggestions"],
    queryFn: () => getMergeSuggestions(),
    staleTime: 30_000,
  });

  const mutation = useMutation({
    mutationFn: (action: PendingAction) =>
      mergeSuggestion(action.entityType, action.keepId, action.removeId, action.approve),
    onSuccess: (result, action) => {
      setMessage(result.summary);
      if (action.approve) {
        queryClient.invalidateQueries({ queryKey: ["merge-suggestions"] });
        queryClient.invalidateQueries({ queryKey: ["inventory"] });
      } else {
        const key = `${action.entityType}:${action.keepId}:${action.removeId}`;
        setDismissed((prev) => new Set(prev).add(key));
      }
    },
  });

  const sections = useMemo(
    () => [
      { title: "Producers", entityType: "producer" as const, suggestions: data?.producers ?? [] },
      { title: "Regions", entityType: "region" as const, suggestions: data?.regions ?? [] },
      { title: "Wines", entityType: "wine" as const, suggestions: data?.wines ?? [] },
      { title: "Possible Wine Matches", entityType: "wine" as const, suggestions: data?.possible_wines ?? [] },
    ]
      .map((section) => ({
        ...section,
        suggestions: section.suggestions.filter(
          (s) => !dismissed.has(`${section.entityType}:${s.keep_id}:${s.remove_id}`),
        ),
      }))
      .filter((section) => section.suggestions.length > 0),
    [data, dismissed],
  );

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-border bg-card px-4 py-3 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        Scanning for duplicate records...
      </div>
    );
  }

  if (isError) {
    return (
      <div className="space-y-3 rounded-lg border border-destructive/20 bg-destructive/5 p-4">
        <p className="text-sm text-destructive">{(error as Error).message ?? "Failed to load merge suggestions."}</p>
        <Button size="sm" variant="outline" onClick={() => refetch()}>
          Retry
        </Button>
      </div>
    );
  }

  const visibleTotal = sections.reduce((acc, section) => acc + section.suggestions.length, 0);

  if (!data || visibleTotal === 0) {
    return (
      <EmptyState
        icon={CheckCircle2}
        title="No merge suggestions"
        description="No visible producer, region, wine, or possible wine matches remain in this session."
      />
    );
  }

  const activeKey = mutation.isPending
    ? `${mutation.variables.entityType}:${mutation.variables.keepId}:${mutation.variables.removeId}:${mutation.variables.approve}`
    : null;

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-amber-300/60 bg-amber-50/40 px-4 py-3 text-sm text-amber-900">
        <div className="flex items-start gap-2">
          <AlertTriangle className="mt-0.5 size-4 shrink-0" />
          <div>
            <p className="font-medium">Development-only manual merge tool</p>
            <p className="text-xs text-amber-900/80">
              Review each suggestion carefully. YES performs permanent database changes.
            </p>
          </div>
        </div>
      </div>

      {message && (
        <div className="rounded-lg border border-emerald-300/60 bg-emerald-50/50 px-4 py-3 text-sm text-emerald-900">
          {message}
        </div>
      )}

      {mutation.isError && (
        <div className="rounded-lg border border-destructive/20 bg-destructive/5 px-4 py-3 text-sm text-destructive">
          {(mutation.error as Error).message ?? "Could not apply merge action."}
        </div>
      )}

      <div className="rounded-lg border border-border bg-card px-4 py-3 text-sm text-muted-foreground">
        Total suggestions: <span className="font-medium text-foreground">{visibleTotal}</span>
      </div>

      {sections.map((section) => (
        <div key={section.title} className="rounded-lg border border-border bg-card">
          <div className="flex items-center justify-between border-b border-border px-4 py-3">
            <h3 className="text-sm font-semibold">{section.title}</h3>
            <span className="text-xs text-muted-foreground">{section.suggestions.length}</span>
          </div>

          {section.suggestions.length === 0 ? (
            <p className="px-4 py-4 text-sm text-muted-foreground">No suggestions in this category.</p>
          ) : (
            <div className="divide-y divide-border">
              {section.suggestions.map((suggestion) => {
                const yesKey = `${section.entityType}:${suggestion.keep_id}:${suggestion.remove_id}:true`;
                const noKey = `${section.entityType}:${suggestion.keep_id}:${suggestion.remove_id}:false`;
                const isBusy = activeKey === yesKey || activeKey === noKey;
                return (
                  <SuggestionRow
                    key={`${section.entityType}-${suggestion.keep_id}-${suggestion.remove_id}`}
                    suggestion={suggestion}
                    isBusy={isBusy}
                    onApprove={() =>
                      mutation.mutate({
                        entityType: section.entityType,
                        keepId: suggestion.keep_id,
                        removeId: suggestion.remove_id,
                        approve: true,
                      })
                    }
                    onReject={() =>
                      mutation.mutate({
                        entityType: section.entityType,
                        keepId: suggestion.keep_id,
                        removeId: suggestion.remove_id,
                        approve: false,
                      })
                    }
                  />
                );
              })}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function SuggestionRow({
  suggestion,
  onApprove,
  onReject,
  isBusy,
}: {
  suggestion: MergeSuggestion;
  onApprove: () => void;
  onReject: () => void;
  isBusy: boolean;
}) {
  return (
    <div className="space-y-3 px-4 py-3">
      <p className="text-xs text-muted-foreground">{suggestion.reason}</p>
      <div className="grid gap-2 md:grid-cols-[1fr,1fr,auto] md:items-center">
        <div className="rounded-md border border-border bg-muted/30 px-3 py-2 text-sm">
          <p className="text-xs text-muted-foreground">Keep</p>
          <p className="font-medium">{suggestion.keep_label}</p>
        </div>
        <div className="rounded-md border border-border bg-muted/30 px-3 py-2 text-sm">
          <p className="text-xs text-muted-foreground">Merge and delete</p>
          <p className="font-medium">{suggestion.remove_label}</p>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" disabled={isBusy} onClick={onApprove}>
            {isBusy ? <Loader2 className="size-4 animate-spin" /> : <CheckCircle2 className="size-4" />} YES
          </Button>
          <Button size="sm" variant="outline" disabled={isBusy} onClick={onReject}>
            <XCircle className="size-4" /> NO
          </Button>
        </div>
      </div>
    </div>
  );
}


