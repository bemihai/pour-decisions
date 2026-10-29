"use client";

import { type KeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useInfiniteQuery, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronDown, Send, Sparkles } from "lucide-react";

import AgentExecutionTimeline from "@/components/AgentExecutionTimeline";
import ChatMessage from "@/components/ChatMessage";
import LogoMark from "@/components/LogoMark";
import { ApiError, createConversation, getConversation, getConversationOperation, getConversationTurns, sendChatMessage, streamChatMessage } from "@/lib/api";
import { CHAT_COMMANDS, resolveChatCommand, type ChatCommand } from "@/lib/chat-commands";
import { cn } from "@/lib/utils";
import { useChatStore, type Message } from "@/stores/chat-store";
import { readLegacyBrowserConversation, useConversationStore } from "@/stores/conversation-store";
import type { AgentMode, ChatRequest, ChatResponse, ConversationMetadata, ConversationOperation, ConversationTurn, ToolProgressStreamEvent } from "@/lib/types";

const STARTERS = ["What wine pairs well with grilled salmon?", "Show me my cellar stats", "Recommend a Pinot Noir under $30", "What are the best wine regions in France?"];
const WELCOME: Message = { role: "ai", content: "Welcome to Pour Decisions! I'm your wine assistant. Ask me anything about wine, your cellar, food pairings, or recommendations." };
interface DisplayMessage extends Message { executionContent?: string; turnSequence?: number }

function setConversationUrl(id: string | null): void {
  const url = new URL(window.location.href);
  if (id) url.searchParams.set("conversation", id); else url.searchParams.delete("conversation");
  window.history.pushState(null, "", `${url.pathname}${url.search}${url.hash}`);
}

function turnMessages(turn: ConversationTurn, mode: AgentMode): DisplayMessage[] {
  const human = { role: "human" as const, content: turn.display_message, executionContent: turn.message, turnSequence: turn.sequence };
  if (turn.status === "completed" && turn.answer !== null) return [human, { role: "ai", content: turn.answer, sources: turn.sources, webSources: turn.web_sources, agentMode: mode, turnSequence: turn.sequence }];
  const text = turn.status === "running"
    ? "This response is still being delivered. Refresh to check its latest state."
    : turn.status === "failed"
      ? "This response failed. Your question remains saved; send a new message when you are ready."
      : "This response was interrupted and was not replayed automatically.";
  return [human, { role: "ai", content: text, isError: true, turnSequence: turn.sequence }];
}

function Thinking() {
  return <div className="mb-4 flex items-start gap-2"><div className="flex h-10 w-10 items-center justify-center rounded-full bg-brand-burgundy/10"><LogoMark size={20} className="text-brand-burgundy" title="" /></div><div role="status" className="rounded-[20px_20px_20px_4px] bg-chat-ai px-4 py-3 text-sm">Thinking…</div></div>;
}

export default function ChatInterface() {
  const queryClient = useQueryClient();
  const selectedId = useConversationStore((s) => s.selectedConversationId);
  const mode = useConversationStore((s) => s.selectedMode);
  const legacySelected = useConversationStore((s) => s.isLegacySelected);
  const selectConversation = useConversationStore((s) => s.selectConversation);
  const beginDraft = useConversationStore((s) => s.beginDraft);
  const isLoading = useChatStore((s) => s.isLoading);
  const setLoading = useChatStore((s) => s.setLoading);
  const [hydrated, setHydrated] = useState(false);
  const [input, setInput] = useState("");
  const [local, setLocal] = useState<DisplayMessage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [guidance, setGuidance] = useState<string | null>(null);
  const [help, setHelp] = useState(false);
  const [menuDismissed, setMenuDismissed] = useState(false);
  const [activeCommand, setActiveCommand] = useState(0);
  const [progress, setProgress] = useState<Map<number, ToolProgressStreamEvent>>(new Map());
  const [showScroll, setShowScroll] = useState(false);
  const composer = useRef<HTMLTextAreaElement>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const request = useRef<AbortController | null>(null);
  const selectedRef = useRef(selectedId);
  const olderHeight = useRef<number | null>(null);

  const metadata = useQuery({ queryKey: ["conversation", selectedId], queryFn: () => getConversation(selectedId!), enabled: hydrated && selectedId !== null && !legacySelected, retry: false });
  const turns = useInfiniteQuery({
    queryKey: ["conversation-turns", selectedId],
    queryFn: ({ pageParam }) => getConversationTurns(selectedId!, 50, pageParam),
    initialPageParam: null as number | null,
    getNextPageParam: (page) => page.next_before_sequence ?? undefined,
    enabled: hydrated && selectedId !== null && !legacySelected,
    retry: false,
  });
  const serverMessages = useMemo(() => [...(turns.data?.pages ?? [])].reverse().flatMap((page) => page.items.flatMap((turn) => turnMessages(turn, mode))), [turns.data, mode]);
  const legacyMessages = useMemo(() => legacySelected ? readLegacyBrowserConversation()?.messages ?? [] : [], [legacySelected]);
  const messages: DisplayMessage[] = legacySelected ? legacyMessages : selectedId === null ? [WELCOME, ...local] : [...serverMessages, ...local];
  const commandQuery = input.trimStart().toLowerCase();
  const commands = CHAT_COMMANDS.filter((item) => item.name.startsWith(commandQuery));
  const showMenu = mode === "intelligent" && !isLoading && !legacySelected && !menuDismissed && commandQuery.startsWith("/") && commands.length > 0;

  useEffect(() => {
    if (useConversationStore.persist.hasHydrated()) setHydrated(true);
    else return useConversationStore.persist.onFinishHydration(() => setHydrated(true));
  }, []);
  useEffect(() => {
    if (!hydrated) return;
    const syncUrl = () => {
      const id = new URL(window.location.href).searchParams.get("conversation");
      if (id) selectConversation(id, useConversationStore.getState().selectedMode);
      else if (!useConversationStore.getState().isLegacySelected) beginDraft();
    };
    syncUrl(); window.addEventListener("popstate", syncUrl);
    return () => window.removeEventListener("popstate", syncUrl);
  }, [hydrated, beginDraft, selectConversation]);
  useEffect(() => { if (metadata.data && metadata.data.agent_mode !== mode) selectConversation(metadata.data.id, metadata.data.agent_mode); }, [metadata.data, mode, selectConversation]);
  useEffect(() => {
    if (selectedRef.current === selectedId) return;
    selectedRef.current = selectedId; request.current?.abort(); request.current = null;
    setLoading(false); setLocal([]); setError(null); setProgress(new Map());
  }, [selectedId, setLoading]);
  useEffect(() => {
    if (olderHeight.current === null || !scroller.current) return;
    scroller.current.scrollTop += scroller.current.scrollHeight - olderHeight.current; olderHeight.current = null;
  }, [turns.data?.pages.length]);
  useEffect(() => { if (hydrated && olderHeight.current === null) bottom.current?.scrollIntoView({ behavior: "smooth" }); }, [hydrated, messages.length, isLoading, progress]);
  useEffect(() => { const el = composer.current; if (el) { el.style.height = "auto"; el.style.height = `${Math.min(el.scrollHeight, 120)}px`; } }, [input]);
  useEffect(() => { const el = scroller.current; if (!el) return; const onScroll = () => setShowScroll(el.scrollHeight - el.scrollTop - el.clientHeight > 120); el.addEventListener("scroll", onScroll, { passive: true }); return () => el.removeEventListener("scroll", onScroll); }, []);
  useEffect(() => () => request.current?.abort(), []);
  useEffect(() => {
    const focusComposer = (event: globalThis.KeyboardEvent) => {
      if (event.key !== "/" || event.ctrlKey || event.metaKey || legacySelected || event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
      event.preventDefault(); composer.current?.focus();
    };
    document.addEventListener("keydown", focusComposer);
    return () => document.removeEventListener("keydown", focusComposer);
  }, [legacySelected]);

  const refresh = useCallback(async (conversationId: string) => {
    await Promise.all([queryClient.invalidateQueries({ queryKey: ["conversations"] }), queryClient.invalidateQueries({ queryKey: ["conversation", conversationId] }), queryClient.invalidateQueries({ queryKey: ["conversation-turns", conversationId] })]);
    setLocal([]);
  }, [queryClient]);
  const applyResponse = useCallback(async (response: ChatResponse) => {
    setLocal((current) => [...current, { role: "ai", content: response.answer, sources: response.sources, webSources: response.web_sources, agentMode: response.agent_mode, isError: response.error !== null }]);
    if (response.conversation_id) await refresh(response.conversation_id);
  }, [refresh]);
  const reconcile = useCallback(async (conversationId: string, operationId: string): Promise<ConversationOperation | null> => {
    try { return await getConversationOperation(conversationId, operationId); } catch { return null; }
  }, []);
  const runRequest = useCallback(async (body: ChatRequest) => {
    const controller = new AbortController(); request.current?.abort(); request.current = controller; setProgress(new Map());
    try {
      return body.agent_mode === "intelligent"
        ? await streamChatMessage(body, { signal: controller.signal, onProgress: (event) => { if (request.current === controller) setProgress((current) => new Map(current).set(event.invocation_id, event)); } })
        : await sendChatMessage(body);
    } finally { if (request.current === controller) request.current = null; setProgress(new Map()); }
  }, []);
  const handleFailure = useCallback(async (caught: unknown, conversationId: string, operationId: string) => {
    if (caught instanceof ApiError && caught.status === 409 && caught.details?.current) {
      queryClient.setQueryData(["conversation", conversationId], caught.details.current);
      setError("This conversation changed elsewhere. Its latest saved state has been loaded; review it before sending again."); await refresh(conversationId); return;
    }
    const operation = await reconcile(conversationId, operationId);
    if (operation?.status === "completed" && operation.response) { await applyResponse(operation.response); return; }
    await refresh(conversationId);
    setError(operation?.status === "failed" ? "The response failed and was not retried automatically. Your question remains saved." : operation?.status === "interrupted" ? "The response was interrupted and was not replayed automatically." : "The request outcome is uncertain. Refresh to check its saved state; it was not replayed automatically.");
  }, [applyResponse, queryClient, reconcile, refresh]);

  const submit = useCallback(async (text: string) => {
    if (!text.trim() || isLoading || legacySelected) return;
    const display = text.trim(); const resolved = resolveChatCommand(display, mode);
    setGuidance(null); setHelp(false);
    if (resolved.kind === "guidance") { setGuidance(resolved.message); return; }
    if (resolved.kind === "help") { setHelp(true); setMenuDismissed(true); setInput(""); return; }
    setInput(""); setError(null); setLoading(true);
    let id = selectedId; let current: ConversationMetadata | undefined = metadata.data;
    try {
      if (id === null) { id = crypto.randomUUID(); current = await createConversation({ id, agent_mode: mode, first_message: display }); selectedRef.current = id; selectConversation(id, mode); setConversationUrl(id); }
      const operationId = crypto.randomUUID();
      setLocal((items) => [...items, { role: "human", content: display, executionContent: resolved.message }]);
      try {
        const response = await runRequest({ message: resolved.message, display_message: display, agent_mode: mode, thread_id: id, thread_action: "append", conversation_id: id, operation_id: operationId, expected_version: current?.version ?? 1 });
        await applyResponse(response);
      } catch (caught) { await handleFailure(caught, id, operationId); }
    } catch (caught) { setError(`Could not start the saved conversation: ${caught instanceof Error ? caught.message : "An unexpected error occurred."}`); }
    finally { setLoading(false); composer.current?.focus(); }
  }, [applyResponse, handleFailure, isLoading, legacySelected, metadata.data, mode, runRequest, selectedId, selectConversation, setLoading]);

  const regenerate = useCallback(async () => {
    if (isLoading || !selectedId || !metadata.data) return;
    const last = [...(turns.data?.pages ?? [])].flatMap((page) => page.items).filter((turn) => turn.status === "completed").sort((a, b) => b.sequence - a.sequence)[0];
    if (!last) return;
    const operationId = crypto.randomUUID(); setError(null); setLoading(true);
    try {
      const response = await runRequest({ message: last.message, display_message: last.display_message, agent_mode: mode, thread_id: selectedId, thread_action: "replace_last", conversation_id: selectedId, operation_id: operationId, expected_version: metadata.data.version });
      await applyResponse(response);
    } catch (caught) { await handleFailure(caught, selectedId, operationId); }
    finally { setLoading(false); composer.current?.focus(); }
  }, [applyResponse, handleFailure, isLoading, metadata.data, mode, runRequest, selectedId, setLoading, turns.data?.pages]);

  function chooseCommand(command: ChatCommand): void { const prefix = `${command.name}${command.hint ? " " : ""}`; setInput(prefix + command.hint); setMenuDismissed(true); requestAnimationFrame(() => composer.current?.focus()); }
  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>): void {
    if (event.nativeEvent.isComposing) return;
    if (showMenu && event.key === "Escape") { event.preventDefault(); setMenuDismissed(true); return; }
    if (showMenu && (event.key === "ArrowDown" || event.key === "ArrowUp")) { event.preventDefault(); setActiveCommand((value) => (value + (event.key === "ArrowDown" ? 1 : -1) + commands.length) % commands.length); return; }
    if (showMenu && ((event.key === "Enter" && !event.shiftKey) || event.key === "Tab")) { event.preventDefault(); chooseCommand(commands[activeCommand] ?? commands[0]); return; }
    if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submit(input); }
  }
  async function loadOlder(): Promise<void> { if (!scroller.current) return; olderHeight.current = scroller.current.scrollHeight; await turns.fetchNextPage(); }

  if (!hydrated) return <div className="flex h-full items-center justify-center"><div className="h-10 w-10 animate-pulse rounded-full bg-muted" /></div>;
  const transcriptLoading = selectedId !== null && (metadata.isPending || turns.isPending);
  const transcriptError = selectedId !== null && (metadata.isError || turns.isError);
  const notFound = metadata.error instanceof ApiError && metadata.error.status === 404;
  const lastAi = messages.findLastIndex((message) => message.role === "ai" && !message.isError);

  return <div className="relative flex h-full flex-col">
    <div ref={scroller} className="flex-1 overflow-y-auto px-6 py-4" role="log" aria-label="Chat messages" aria-live="polite"><div className="mx-auto max-w-5xl">
      {legacySelected && <div role="status" className="mb-4 rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 text-sm">Legacy browser chat — read only. It remains only in this browser and is never uploaded or reconstructed.</div>}
      {transcriptLoading && <p role="status" className="py-8 text-center text-sm text-muted-foreground">Loading saved conversation…</p>}
      {transcriptError && <div role="alert" className="my-6 rounded-lg border border-destructive/30 p-4 text-sm"><p>{notFound ? "This saved conversation was not found." : "Could not load this saved conversation."}</p><button className="mt-2 underline" onClick={() => { metadata.refetch(); turns.refetch(); }}>Try again</button></div>}
      {!transcriptLoading && !transcriptError && turns.hasNextPage && <div className="mb-4 text-center"><button disabled={turns.isFetchingNextPage} onClick={loadOlder} className="text-sm text-brand-burgundy underline disabled:opacity-50">{turns.isFetchingNextPage ? "Loading older turns…" : "Load older turns"}</button></div>}
      {!transcriptLoading && !transcriptError && selectedId !== null && messages.length === 0 && <p className="py-8 text-center text-sm text-muted-foreground">This saved conversation has no turns yet.</p>}
      {!transcriptError && messages.map((message, index) => <ChatMessage key={`${message.turnSequence ?? "local"}-${index}`} role={message.role} content={message.content} sources={message.sources} webSources={message.webSources} agentMode={message.agentMode} isError={message.isError} onRegenerate={index === lastAi && selectedId !== null && !isLoading && !legacySelected ? regenerate : undefined} />)}
      {selectedId === null && messages.length === 1 && !legacySelected && !isLoading && <div className="flex flex-col items-center gap-4 py-8"><Sparkles className="h-8 w-8 text-brand-burgundy opacity-50" /><p className="text-sm text-muted-foreground">Try one of these to get started:</p><div className="flex flex-wrap justify-center gap-2">{STARTERS.map((starter) => <button key={starter} onClick={() => submit(starter)} className="rounded-full border px-4 py-2 text-xs">{starter}</button>)}</div></div>}
      {isLoading && progress.size === 0 && <Thinking />}{isLoading && <AgentExecutionTimeline events={[...progress.values()].sort((a, b) => a.invocation_id - b.invocation_id)} />}<div ref={bottom} />
    </div></div>
    {showScroll && <button onClick={() => bottom.current?.scrollIntoView({ behavior: "smooth" })} aria-label="Scroll to latest message" className="absolute bottom-20 right-8 flex h-9 w-9 items-center justify-center rounded-full bg-brand-burgundy text-white"><ChevronDown className="h-4 w-4" /></button>}
    <div className="shrink-0 border-t bg-background/95 px-6 py-3">
      {error && <p role="alert" className="mx-auto mb-2 max-w-5xl text-sm text-destructive">{error}</p>}{guidance && <p id="command-guidance" role="alert" className="mx-auto mb-2 max-w-5xl text-sm text-destructive">{guidance}</p>}
      {help && <div role="region" aria-label="Slash command help" className="mx-auto mb-3 max-w-5xl rounded-lg border p-3 text-sm"><p className="font-medium">Intelligent chat commands</p><ul>{CHAT_COMMANDS.map((command) => <li key={command.name}><code>{command.name} {command.hint}</code> — {command.purpose}</li>)}</ul></div>}
      {showMenu && <ul role="listbox" aria-label="Slash commands" className="mx-auto mb-2 max-w-5xl rounded-lg border bg-background p-1">{commands.map((command, index) => <li key={command.name} role="option" aria-selected={index === activeCommand} onMouseDown={(event) => event.preventDefault()} onClick={() => chooseCommand(command)} className={cn("cursor-pointer rounded px-3 py-2 text-sm", index === activeCommand && "bg-muted")}><span className="font-medium">{command.name} {command.hint}</span><span className="ml-3 text-muted-foreground">{command.purpose}</span></li>)}</ul>}
      {legacySelected ? <p className="text-center text-sm text-muted-foreground">Start a new conversation to continue chatting.</p> : <form onSubmit={(event) => { event.preventDefault(); submit(input); }} className="mx-auto flex max-w-5xl items-end gap-2"><textarea ref={composer} data-chat-composer value={input} onChange={(event) => { setInput(event.target.value); setActiveCommand(0); setMenuDismissed(false); setGuidance(null); setHelp(false); }} onKeyDown={onKeyDown} aria-label="Chat message" aria-describedby={guidance ? "command-guidance" : undefined} placeholder="Ask about wine, your cellar, pairings…" disabled={isLoading || transcriptLoading || transcriptError} autoFocus rows={1} style={{ resize: "none", overflowY: "hidden" }} className="flex-1 rounded-2xl border bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-brand-burgundy disabled:opacity-50" /><button type="submit" disabled={isLoading || transcriptLoading || transcriptError || !input.trim()} aria-label="Send message" className="flex h-10 w-10 items-center justify-center rounded-full bg-brand-burgundy text-white disabled:opacity-50"><Send className="h-4 w-4" /></button></form>}
    </div>
  </div>;
}
