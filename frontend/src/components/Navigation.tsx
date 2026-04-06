"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { MessageCircle, Wine, Palette } from "lucide-react";
import { cn } from "@/lib/utils";
import ThemeToggle from "@/components/ThemeToggle";

const NAV_ITEMS = [
  { href: "/", label: "Chatbot", icon: MessageCircle },
  { href: "/cellar", label: "Cellar", icon: Wine },
  { href: "/taste-profile", label: "Taste Profile", icon: Palette },
] as const;

export default function Navigation() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-6 px-4 sm:px-6">
        {/* Brand title */}
        <Link href="/" className="mr-4 flex shrink-0 items-center gap-2">
          <span className="text-lg font-bold bg-gradient-to-r from-purple-800 via-purple-600 to-purple-700 bg-clip-text text-transparent">
            Pour Decisions
          </span>
        </Link>

        {/* Nav links */}
        <nav className="flex items-center gap-1">
          {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
            const isActive = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-brand-purple text-white"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )}
              >
                <Icon className="h-4 w-4" />
                {label}
              </Link>
            );
          })}
        </nav>

        {/* Theme toggle pushed to the right */}
        <div className="ml-auto">
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}

