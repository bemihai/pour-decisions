/** Responsive conversation library and fixed-mode controls for the chat page. */
"use client";

import { useState } from "react";
import { BookOpen, Brain, Library } from "lucide-react";

import ConversationList from "@/components/chat/ConversationList";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { useChatStore } from "@/stores/chat-store";
import { useConversationStore } from "@/stores/conversation-store";
import type { AgentMode } from "@/lib/types";

const AGENT_OPTIONS: Array<{
  value: AgentMode;
  label: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
}> = [
  {
    value: "intelligent",
    label: "Intelligent Agent",
    description: "Uses tools for complex cellar and wine requests.",
    icon: Brain,
  },
  {
    value: "rag_only",
    label: "RAG Only",
    description: "Uses bounded saved history and wine retrieval.",
    icon: BookOpen,
  },
];

function clearConversationUrl(): void {
  const url = new URL(window.location.href);
  url.searchParams.delete("conversation");
  window.history.pushState(null, "", `${url.pathname}${url.search}${url.hash}`);
}

function SidebarContent() {
  const isLoading = useChatStore((state) => state.isLoading);
  const selectedConversationId = useConversationStore((state) => state.selectedConversationId);
  const selectedMode = useConversationStore((state) => state.selectedMode);
  const beginDraft = useConversationStore((state) => state.beginDraft);

  function changeMode(mode: AgentMode): void {
    if (isLoading || (mode === selectedMode && selectedConversationId === null)) return;
    beginDraft(mode);
    clearConversationUrl();
    requestAnimationFrame(() => {
      document.querySelector<HTMLTextAreaElement>("[data-chat-composer]")?.focus();
    });
  }

  return (
    <div className="flex flex-col gap-5">
      <ConversationList />
      <hr className="border-border" />
      <section aria-labelledby="agent-mode-heading" className="flex flex-col gap-2">
        <h2
          id="agent-mode-heading"
          className="text-xs font-semibold uppercase tracking-wide text-muted-foreground"
        >
          Mode for new conversation
        </h2>
        {AGENT_OPTIONS.map(({ value, label, description, icon: Icon }) => {
          const active = selectedMode === value;
          return (
            <button
              key={value}
              type="button"
              onClick={() => changeMode(value)}
              disabled={isLoading}
              aria-pressed={active}
              className={cn(
                "rounded-lg border px-3 py-2 text-left transition-colors",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50",
                active
                  ? "border-brand-burgundy bg-brand-burgundy/5"
                  : "border-border hover:bg-muted/60",
              )}
            >
              <span className="flex items-center gap-2 text-sm font-medium">
                <Icon className="h-4 w-4" />
                {label}
              </span>
              <span className="mt-0.5 block pl-6 text-[11px] leading-snug text-muted-foreground">
                {description}
              </span>
            </button>
          );
        })}
        {selectedConversationId !== null && (
          <p className="text-[11px] text-muted-foreground">
            Saved conversation modes are fixed. Choosing a mode starts a new draft.
          </p>
        )}
      </section>
    </div>
  );
}

function DesktopSidebar() {
  return (
    <aside className="hidden w-72 shrink-0 overflow-y-auto border-l border-border bg-muted/30 px-4 py-5 lg:flex lg:flex-col xl:w-80">
      <SidebarContent />
    </aside>
  );
}

function MobileSidebarTrigger() {
  const [open, setOpen] = useState(false);
  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <SheetTrigger asChild>
              <button
                aria-label="Open conversation library"
                className={cn(
                  "flex h-9 w-9 items-center justify-center rounded-md border border-border lg:hidden",
                  "text-muted-foreground transition-colors hover:bg-muted hover:text-foreground",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                )}
              >
                <Library className="h-4 w-4" />
              </button>
            </SheetTrigger>
          </TooltipTrigger>
          <TooltipContent side="bottom">Conversations</TooltipContent>
        </Tooltip>
      </TooltipProvider>
      <SheetContent side="right" className="w-80 overflow-y-auto">
        <SheetHeader>
          <SheetTitle>Conversations</SheetTitle>
        </SheetHeader>
        <div className="px-1 pt-4">
          <SidebarContent />
        </div>
      </SheetContent>
    </Sheet>
  );
}

export default DesktopSidebar;
export { MobileSidebarTrigger };
