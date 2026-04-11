/**
 * Breadcrumbs component.
 *
 * Renders a horizontal navigation trail with chevron separators.
 * Linked items navigate to their href; the last item is non-linked
 * (current page) and displayed with foreground weight.
 *
 * Usage:
 *   <Breadcrumbs items={[
 *     { label: "Cellar", href: "/cellar" },
 *     { label: "Château Margaux", href: "/cellar?producer=..." },
 *     { label: "Château Margaux 2015" },
 *   ]} />
 */

import Link from "next/link";
import { ChevronRight } from "lucide-react";

import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface BreadcrumbItem {
  label: string;
  href?: string;
}

export interface BreadcrumbsProps {
  items: BreadcrumbItem[];
  className?: string;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function Breadcrumbs({ items, className }: BreadcrumbsProps) {
  return (
    <nav
      aria-label="Breadcrumb"
      className={cn("flex flex-wrap items-center gap-1 type-caption text-muted-foreground", className)}
    >
      {items.map((item, i) => (
        <span key={i} className="flex items-center gap-1">
          {i > 0 && <ChevronRight className="size-3.5 shrink-0" aria-hidden="true" />}
          {item.href ? (
            <Link
              href={item.href}
              className="max-w-[200px] truncate hover:text-foreground transition-colors"
            >
              {item.label}
            </Link>
          ) : (
            <span className="max-w-[240px] truncate font-medium text-foreground" aria-current="page">
              {item.label}
            </span>
          )}
        </span>
      ))}
    </nav>
  );
}

