/**
 * ChatSidebar component.
 *
 * Agent mode selection panel and reset control for the chatbot page.
 * Replaces render_sidebar() from src/ui/sidebar.py.
 *
 * Desktop (lg+): static aside on the right.
 * Mobile (< lg): accessible via a Sheet (drawer) triggered from the page header.
 * Client Component — uses useChatStore for shared state.
 */
"use client";

import { useState } from "react";
import { Brain, Zap, BookOpen, RotateCcw, Settings2, Monitor, Cloud } from "lucide-react";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";

import { useChatStore } from "@/stores/chat-store";
import { cn } from "@/lib/utils";
import type { AgentMode, ModelProvider } from "@/lib/types";

// ---------------------------------------------------------------------------
// Agent mode config
// ---------------------------------------------------------------------------

interface AgentOption {
  value: AgentMode;
  label: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
}

const AGENT_OPTIONS: AgentOption[] = [
  {
    value: "intelligent",
    label: "Intelligent Agent",
    description:
      "Uses LLM to intelligently select and chain tools. Best for complex queries.",
    icon: Brain,
  },
  {
    value: "keyword",
    label: "Keyword Agent",
    description:
      "Uses pattern matching for routing. Faster, uses fewer LLM calls, ideal for testing.",
    icon: Zap,
  },
  {
    value: "rag_only",
    label: "No Agent (RAG Only)",
    description:
      "Traditional RAG without agents. Uses only wine knowledge retrieval.",
    icon: BookOpen,
  },
];

// ---------------------------------------------------------------------------
// Model provider config
// ---------------------------------------------------------------------------

interface ModelOption {
  value: ModelProvider;
  label: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
}

const MODEL_OPTIONS: ModelOption[] = [
  {
    value: "local",
    label: "Local (Gemma 4)",
    description: "Runs on your machine via Ollama. Free, private, no API calls.",
    icon: Monitor,
  },
  {
    value: "cloud",
    label: "Cloud (Gemini)",
    description: "Google Gemini API. Faster, more capable, uses API quota.",
    icon: Cloud,
  },
];

// ---------------------------------------------------------------------------
// Shared sidebar content (reused in desktop aside and mobile Sheet)
// ---------------------------------------------------------------------------

function SidebarContent() {
  const { agentMode, setAgentMode, modelProvider, setModelProvider, resetChat, isLoading } = useChatStore();

  return (
    <div className="flex flex-col gap-6">
      {/* App description */}
      <p className="text-sm text-muted-foreground leading-relaxed">
        Pour Decisions uses Retrieval-Augmented Generation (RAG) and LLMs to
        answer your wine-related questions using curated knowledge and your
        personal cellar data.
      </p>

      <hr className="border-border" />

      {/* Agent mode selection */}
      <div className="flex flex-col gap-3">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Agent Mode
        </h2>

        {AGENT_OPTIONS.map(({ value, label, description, icon: Icon }) => {
          const isActive = agentMode === value;
          return (
            <button
              key={value}
              onClick={() => setAgentMode(value)}
              disabled={isLoading}
              className={cn(
                "group w-full text-left rounded-lg border px-3 py-3 transition-all",
                "focus:outline-none focus:ring-2 focus:ring-brand-burgundy focus:ring-offset-1",
                "disabled:opacity-50 disabled:cursor-not-allowed",
                isActive
                  ? "border-brand-burgundy ring-2 ring-brand-burgundy bg-background"
                  : "border-border bg-background hover:border-brand-burgundy/50 hover:bg-muted/50",
              )}
              aria-pressed={isActive}
            >
              <div className="flex items-center gap-2">
                <Icon
                  className={cn(
                    "h-4 w-4 shrink-0",
                    isActive ? "text-brand-burgundy" : "text-muted-foreground",
                  )}
                />
                <span
                  className={cn(
                    "text-sm font-medium",
                    isActive ? "text-brand-burgundy" : "text-foreground",
                  )}
                >
                  {label}
                </span>
              </div>
              <p className="text-xs text-muted-foreground leading-snug pl-6 mt-1 hidden group-hover:block group-focus-visible:block">
                {description}
              </p>
            </button>
          );
        })}
      </div>

      <hr className="border-border" />

      {/* Model provider selection */}
      <div className="flex flex-col gap-3">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Model
        </h2>

        {MODEL_OPTIONS.map(({ value, label, description, icon: Icon }) => {
          const isActive = modelProvider === value;
          return (
            <button
              key={value}
              onClick={() => setModelProvider(value)}
              disabled={isLoading}
              className={cn(
                "w-full text-left rounded-lg border px-3 py-3 transition-all",
                "focus:outline-none focus:ring-2 focus:ring-brand-burgundy focus:ring-offset-1",
                "disabled:opacity-50 disabled:cursor-not-allowed",
                isActive
                  ? "border-brand-burgundy ring-2 ring-brand-burgundy bg-background"
                  : "border-border bg-background hover:border-brand-burgundy/50 hover:bg-muted/50",
              )}
              aria-pressed={isActive}
            >
              <div className="flex items-center gap-2 mb-1">
                <Icon
                  className={cn(
                    "h-4 w-4 shrink-0",
                    isActive ? "text-brand-burgundy" : "text-muted-foreground",
                  )}
                />
                <span
                  className={cn(
                    "text-sm font-medium",
                    isActive ? "text-brand-burgundy" : "text-foreground",
                  )}
                >
                  {label}
                </span>
              </div>
              <p className="text-xs text-muted-foreground leading-snug pl-6">
                {description}
              </p>
            </button>
          );
        })}
      </div>

      <hr className="border-border" />

      {/* Reset chat — wrapped in a Dialog for confirmation */}
      <Dialog>
        <DialogTrigger asChild>
          <button
            disabled={isLoading}
            className={cn(
              "flex items-center justify-center gap-2 w-full rounded-lg border border-border",
              "px-3 py-2.5 text-sm font-medium text-muted-foreground",
              "hover:border-destructive hover:text-destructive hover:bg-destructive/5 transition-colors",
              "focus:outline-none focus:ring-2 focus:ring-destructive focus:ring-offset-1",
              "disabled:opacity-50 disabled:cursor-not-allowed",
            )}
          >
            <RotateCcw className="h-4 w-4" />
            Reset Chat
          </button>
        </DialogTrigger>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Reset conversation?</DialogTitle>
            <DialogDescription>
              This will clear all messages from this conversation. This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="outline" size="sm">Cancel</Button>
            </DialogClose>
            <DialogClose asChild>
              <Button variant="destructive" size="sm" onClick={resetChat}>
                Reset Chat
              </Button>
            </DialogClose>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Desktop sidebar (hidden below lg)
// ---------------------------------------------------------------------------

function DesktopSidebar() {
  return (
    <aside className="hidden lg:flex flex-col w-64 xl:w-72 shrink-0 overflow-y-auto border-l border-border bg-muted/30 px-4 py-6">
      <SidebarContent />
    </aside>
  );
}

// ---------------------------------------------------------------------------
// Mobile sidebar trigger + Sheet (B2, visible below lg)
// ---------------------------------------------------------------------------

function MobileSidebarTrigger() {
  const [open, setOpen] = useState(false);

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <SheetTrigger asChild>
              <button
                aria-label="Open chat settings"
                className={cn(
                  "lg:hidden flex h-9 w-9 items-center justify-center rounded-md border border-border",
                  "text-muted-foreground hover:text-foreground hover:bg-muted transition-colors",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                )}
              >
                <Settings2 className="h-4 w-4" />
              </button>
            </SheetTrigger>
          </TooltipTrigger>
          <TooltipContent side="bottom">Chat settings</TooltipContent>
        </Tooltip>
      </TooltipProvider>
      <SheetContent side="right" className="w-80 overflow-y-auto">
        <SheetHeader>
          <SheetTitle>Chat Settings</SheetTitle>
        </SheetHeader>
        <div className="px-1 pt-4">
          <SidebarContent />
        </div>
      </SheetContent>
    </Sheet>
  );
}

// ---------------------------------------------------------------------------
// Public exports
// ---------------------------------------------------------------------------

export default DesktopSidebar;
export { MobileSidebarTrigger };
