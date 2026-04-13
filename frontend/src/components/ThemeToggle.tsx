/**
 * ThemeToggle component.
 *
 * Cycles through system / light / dark themes.
 * Persists the user's choice in localStorage under "pour-decisions-theme".
 * Applies or removes the "dark" class on the <html> element.
 *
 * System preference is the default. The toggle cycles: system -> light -> dark.
 */
"use client";

import { useEffect, useState, useCallback } from "react";
import { Sun, Moon, Monitor } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

type Theme = "system" | "light" | "dark";

const STORAGE_KEY = "pour-decisions-theme";
const CYCLE: Theme[] = ["system", "light", "dark"];

const ICONS: Record<Theme, React.ComponentType<{ className?: string }>> = {
  system: Monitor,
  light: Sun,
  dark: Moon,
};

const LABELS: Record<Theme, string> = {
  system: "System theme",
  light: "Light theme",
  dark: "Dark theme",
};

function applyTheme(theme: Theme) {
  const root = document.documentElement;
  if (theme === "system") {
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    root.classList.toggle("dark", prefersDark);
  } else {
    root.classList.toggle("dark", theme === "dark");
  }
}

export default function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("system");
  const [mounted, setMounted] = useState(false);

  // Initialise from localStorage on mount.
  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY) as Theme | null;
    const initial = stored && CYCLE.includes(stored) ? stored : "system";
    setTheme(initial);
    applyTheme(initial);
    setMounted(true);
  }, []);

  // Listen for system preference changes when in "system" mode.
  useEffect(() => {
    if (theme !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = () => applyTheme("system");
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, [theme]);

  const cycle = useCallback(() => {
    const nextIdx = (CYCLE.indexOf(theme) + 1) % CYCLE.length;
    const next = CYCLE[nextIdx];
    setTheme(next);
    localStorage.setItem(STORAGE_KEY, next);
    applyTheme(next);
  }, [theme]);

  // Avoid hydration mismatch: render a placeholder until mounted.
  if (!mounted) {
    return <div className="h-8 w-8" />;
  }

  const Icon = ICONS[theme];

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            onClick={cycle}
            aria-label={LABELS[theme]}
            className={cn(
              "flex h-8 w-8 items-center justify-center rounded-md",
              "text-muted-foreground hover:text-foreground hover:bg-muted transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
            )}
          >
            <Icon className="h-4 w-4" />
          </button>
        </TooltipTrigger>
        <TooltipContent side="bottom">{LABELS[theme]}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

