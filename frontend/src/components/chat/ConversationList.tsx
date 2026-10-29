"use client";

import { useMemo, useState } from "react";
import { useInfiniteQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Clock3, History, Pencil, Plus, Trash2, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogClose, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { deleteConversation, listConversations, renameConversation } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useChatStore } from "@/stores/chat-store";
import { dismissLegacyBrowserConversation, readLegacyBrowserConversation, useConversationStore } from "@/stores/conversation-store";
import type { ConversationMetadata } from "@/lib/types";

const PAGE_SIZE = 30;

function updateConversationUrl(conversationId: string | null): void {
  const url = new URL(window.location.href);
  if (conversationId === null) url.searchParams.delete("conversation");
  else url.searchParams.set("conversation", conversationId);
  window.history.pushState(null, "", `${url.pathname}${url.search}${url.hash}`);
}

function focusComposer(): void {
  requestAnimationFrame(() => document.querySelector<HTMLTextAreaElement>("[data-chat-composer]")?.focus());
}

function formatActivity(value: string): string {
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(new Date(value));
}

export default function ConversationList() {
  const queryClient = useQueryClient();
  const isLoading = useChatStore((state) => state.isLoading);
  const selectedConversationId = useConversationStore((state) => state.selectedConversationId);
  const isLegacySelected = useConversationStore((state) => state.isLegacySelected);
  const selectedMode = useConversationStore((state) => state.selectedMode);
  const beginDraft = useConversationStore((state) => state.beginDraft);
  const selectConversation = useConversationStore((state) => state.selectConversation);
  const selectLegacy = useConversationStore((state) => state.selectLegacy);
  const [renameTarget, setRenameTarget] = useState<ConversationMetadata | null>(null);
  const [renameTitle, setRenameTitle] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<ConversationMetadata | null>(null);
  const [legacy, setLegacy] = useState(() => readLegacyBrowserConversation());

  const conversations = useInfiniteQuery({
    queryKey: ["conversations"],
    queryFn: ({ pageParam }) => listConversations(PAGE_SIZE, pageParam),
    initialPageParam: null as string | null,
    getNextPageParam: (page) => page.next_cursor ?? undefined,
  });
  const items = useMemo(() => {
    const unique = new Map<string, ConversationMetadata>();
    for (const page of conversations.data?.pages ?? []) {
      for (const item of page.items) unique.set(item.id, item);
    }
    return [...unique.values()];
  }, [conversations.data]);

  const renameMutation = useMutation({
    mutationFn: ({ target, title }: { target: ConversationMetadata; title: string }) =>
      renameConversation(target.id, { title, expected_version: target.version }),
    onSuccess: async () => {
      setRenameTarget(null);
      await queryClient.invalidateQueries({ queryKey: ["conversations"] });
    },
  });
  const deleteMutation = useMutation({
    mutationFn: (target: ConversationMetadata) => deleteConversation(target.id, target.version),
    onSuccess: async (_result, target) => {
      setDeleteTarget(null);
      if (selectedConversationId === target.id) {
        beginDraft(target.agent_mode);
        updateConversationUrl(null);
        focusComposer();
      }
      await queryClient.invalidateQueries({ queryKey: ["conversations"] });
    },
  });
  const controlsDisabled = isLoading || renameMutation.isPending || deleteMutation.isPending;

  function startNew(): void {
    if (controlsDisabled) return;
    beginDraft(selectedMode);
    updateConversationUrl(null);
    focusComposer();
  }

  function openConversation(item: ConversationMetadata): void {
    if (controlsDisabled || item.state !== "active") return;
    selectConversation(item.id, item.agent_mode);
    updateConversationUrl(item.id);
    focusComposer();
  }

  function dismissLegacy(): void {
    dismissLegacyBrowserConversation();
    setLegacy(null);
    if (isLegacySelected) beginDraft(selectedMode);
  }

  return (
    <section aria-labelledby="conversation-library-heading" className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <h2 id="conversation-library-heading" className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Conversations</h2>
        <Button size="sm" onClick={startNew} disabled={controlsDisabled} className="h-8 gap-1.5"><Plus className="h-3.5 w-3.5" /> New</Button>
      </div>
      {conversations.isPending && <p role="status" className="text-sm text-muted-foreground">Loading conversations…</p>}
      {conversations.isError && <div role="alert" className="rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm"><p>Could not load conversations.</p><Button variant="outline" size="sm" className="mt-2" onClick={() => conversations.refetch()}>Retry</Button></div>}
      {!conversations.isPending && !conversations.isError && items.length === 0 && <p className="rounded-lg border border-dashed border-border p-3 text-sm text-muted-foreground">No saved conversations yet. Your first message creates one.</p>}
      <div className="flex flex-col gap-1" aria-label="Recent conversations">
        {items.map((item) => {
          const selected = selectedConversationId === item.id && !isLegacySelected;
          return <div key={item.id} className={cn("flex items-center rounded-lg border", selected ? "border-brand-burgundy bg-brand-burgundy/5" : "border-transparent hover:bg-muted/60")}>
            <button type="button" aria-label={`Open ${item.title}`} onClick={() => openConversation(item)} disabled={controlsDisabled || item.state !== "active"} aria-current={selected ? "page" : undefined} className="min-w-0 flex-1 px-2.5 py-2 text-left disabled:cursor-not-allowed disabled:opacity-60">
              <span className="block truncate text-sm font-medium">{item.title}</span>
              <span className="mt-0.5 flex items-center gap-1.5 text-[11px] text-muted-foreground"><span>{item.agent_mode === "intelligent" ? "Intelligent" : "RAG only"}</span><span aria-hidden>·</span><Clock3 className="h-3 w-3" aria-hidden /><span>{item.state === "deleting" ? "Deleting…" : formatActivity(item.updated_at)}</span></span>
            </button>
            <div className="flex shrink-0 pr-1">
              <button type="button" aria-label={`Rename ${item.title}`} disabled={controlsDisabled || item.state !== "active"} onClick={() => { setRenameTarget(item); setRenameTitle(item.title); }} className="rounded p-1.5 text-muted-foreground hover:bg-background hover:text-foreground disabled:opacity-40"><Pencil className="h-3.5 w-3.5" /></button>
              <button type="button" aria-label={`Delete ${item.title}`} disabled={controlsDisabled} onClick={() => setDeleteTarget(item)} className="rounded p-1.5 text-muted-foreground hover:bg-destructive/10 hover:text-destructive disabled:opacity-40"><Trash2 className="h-3.5 w-3.5" /></button>
            </div>
          </div>;
        })}
      </div>
      {conversations.hasNextPage && <Button variant="ghost" size="sm" disabled={controlsDisabled || conversations.isFetchingNextPage} onClick={() => conversations.fetchNextPage()}>{conversations.isFetchingNextPage ? "Loading…" : "Load more"}</Button>}
      {legacy && <div className={cn("flex items-center rounded-lg border", isLegacySelected ? "border-brand-burgundy bg-brand-burgundy/5" : "border-border")}>
        <button type="button" onClick={() => { if (!controlsDisabled) { selectLegacy(); updateConversationUrl(null); } }} disabled={controlsDisabled} className="min-w-0 flex-1 px-2.5 py-2 text-left"><span className="flex items-center gap-2 text-sm font-medium"><History className="h-3.5 w-3.5" /> Legacy browser chat</span><span className="block text-[11px] text-muted-foreground">Read-only · stored only in this browser</span></button>
        <button type="button" aria-label="Dismiss legacy browser chat" disabled={controlsDisabled} onClick={dismissLegacy} className="mr-1 rounded p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground"><X className="h-3.5 w-3.5" /></button>
      </div>}
      {(renameMutation.isError || deleteMutation.isError) && <p role="alert" className="rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm">{renameMutation.isError ? "Could not rename the conversation. Refresh and try again." : "Could not delete the conversation. Refresh and try again."}</p>}

      <Dialog open={renameTarget !== null} onOpenChange={(open) => !open && setRenameTarget(null)}><DialogContent><DialogHeader><DialogTitle>Rename conversation</DialogTitle><DialogDescription>Choose a short title for this saved conversation.</DialogDescription></DialogHeader><Input aria-label="Conversation title" value={renameTitle} maxLength={120} onChange={(event) => setRenameTitle(event.target.value)} /><DialogFooter><DialogClose asChild><Button variant="outline">Cancel</Button></DialogClose><Button disabled={!renameTitle.trim() || renameMutation.isPending} onClick={() => renameTarget && renameMutation.mutate({ target: renameTarget, title: renameTitle.trim() })}>Save</Button></DialogFooter></DialogContent></Dialog>
      <Dialog open={deleteTarget !== null} onOpenChange={(open) => !open && setDeleteTarget(null)}><DialogContent><DialogHeader><DialogTitle>Delete conversation?</DialogTitle><DialogDescription>This removes the saved transcript and resumable context. It cannot be undone.</DialogDescription></DialogHeader><DialogFooter><DialogClose asChild><Button variant="outline">Cancel</Button></DialogClose><Button variant="destructive" disabled={deleteMutation.isPending} onClick={() => deleteTarget && deleteMutation.mutate(deleteTarget)}>Delete</Button></DialogFooter></DialogContent></Dialog>
    </section>
  );
}
