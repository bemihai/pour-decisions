"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";

/** Wraps the app with TanStack Query. One QueryClient per browser session. */
export default function QueryProvider({ children }: { children: React.ReactNode }) {
  // useState ensures each request gets its own QueryClient in SSR,
  // and the client is not recreated on every render.
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60_000, // 60 s — avoids redundant refetches for mostly-static cellar data
            retry: 1,
          },
        },
      }),
  );

  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

