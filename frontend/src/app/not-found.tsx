/**
 * Global 404 — Not Found page.
 *
 * Next.js renders this for any URL that doesn't match a route. Branded with
 * the Pour Decisions identity and a clear path back to the chatbot.
 */

import Link from "next/link";
import { Wine } from "lucide-react";
import { Button } from "@/components/ui/button";
import LogoMark from "@/components/LogoMark";

export default function NotFound() {
  return (
    <div className="flex min-h-[calc(100vh-3.5rem)] flex-col items-center justify-center gap-8 px-4 text-center">
      {/* Logo mark */}
      <div className="flex items-center justify-center rounded-2xl bg-brand-burgundy p-5 text-white shadow-md">
        <LogoMark size={48} className="text-white" />
      </div>

      {/* Heading */}
      <div className="flex flex-col gap-3">
        <h1 className="type-page-title text-foreground">Page not found</h1>
        <p className="type-body text-muted-foreground max-w-sm">
          Looks like this bottle has already been consumed. The page you&apos;re
          looking for doesn&apos;t exist.
        </p>
      </div>

      {/* Decorative rule */}
      <div className="h-px w-16 bg-brand-gold" />

      {/* Actions */}
      <div className="flex flex-col items-center gap-3 sm:flex-row">
        <Button asChild className="bg-brand-burgundy text-white hover:bg-brand-burgundy-dark gap-2">
          <Link href="/">
            <Wine className="size-4" />
            Back to Chatbot
          </Link>
        </Button>
        <Button asChild variant="outline">
          <Link href="/cellar">View Cellar</Link>
        </Button>
      </div>

      {/* Subtle 404 label */}
      <p className="type-caption text-muted-foreground/50 select-none">404</p>
    </div>
  );
}

