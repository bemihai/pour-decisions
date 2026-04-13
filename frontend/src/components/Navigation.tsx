"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { MessageCircle, Wine, Palette } from "lucide-react";
import { cn } from "@/lib/utils";
import ThemeToggle from "@/components/ThemeToggle";
import LogoMark from "@/components/LogoMark";

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

        {/* Brand — logo mark + wordmark */}
        <Link
          href="/"
          className="mr-4 flex shrink-0 items-center gap-2.5 group"
          aria-label="Pour Decisions — home"
        >
          <span className="flex items-center justify-center text-brand-burgundy group-hover:text-brand-burgundy-dark transition-colors">
            <LogoMark size={28} title="" aria-hidden="true" />
          </span>
          <span className="text-lg font-bold leading-none tracking-tight bg-gradient-to-r from-brand-burgundy via-brand-burgundy-light to-brand-gold bg-clip-text text-transparent">
            Pour Decisions
          </span>
        </Link>

        {/* Nav links */}
        <nav aria-label="Main navigation" className="flex items-center gap-1">
          {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
            const isActive = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                aria-current={isActive ? "page" : undefined}
                className={cn(
                  "flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-brand-burgundy text-white shadow-sm"
                    : "text-muted-foreground hover:bg-secondary hover:text-foreground",
                )}
              >
                <Icon className="h-4 w-4" aria-hidden="true" />
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



