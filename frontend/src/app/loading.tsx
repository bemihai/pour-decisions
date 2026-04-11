/**
 * Root route (chatbot) loading skeleton.
 *
 * Next.js shows this while the chatbot page resolves. The chat page is a
 * Client Component so this renders only during the initial Server Component
 * shell render — practically instantaneous, but the skeleton prevents any
 * blank-flash if navigation is slow.
 */

function Bone({ className }: { className?: string }) {
  return <div className={`animate-pulse rounded-lg bg-muted ${className ?? ""}`} />;
}

export default function ChatLoading() {
  return (
    <div className="flex flex-col h-[calc(100vh-3.5rem)]">
      {/* Header */}
      <div className="shrink-0 flex items-center justify-between px-6 pt-5 pb-4 border-b border-border">
        <div className="flex flex-col gap-2">
          <Bone className="h-7 w-44" />
          <Bone className="h-4 w-56" />
        </div>
      </div>

      {/* Chat area */}
      <div className="flex flex-1 overflow-hidden">
        <div className="flex flex-1 flex-col gap-4 p-6 overflow-y-auto">
          {/* Welcome bubble */}
          <div className="flex gap-3 max-w-xl">
            <Bone className="size-8 rounded-full shrink-0" />
            <div className="flex flex-col gap-2 flex-1">
              <Bone className="h-4 w-3/4" />
              <Bone className="h-4 w-full" />
              <Bone className="h-4 w-2/3" />
            </div>
          </div>
        </div>

        {/* Sidebar placeholder */}
        <aside className="hidden md:flex flex-col w-[272px] border-l border-border p-4 gap-3">
          <Bone className="h-5 w-28" />
          <Bone className="h-9 w-full" />
          <Bone className="h-9 w-full" />
          <Bone className="h-9 w-full mt-2" />
        </aside>
      </div>

      {/* Input area */}
      <div className="shrink-0 border-t border-border px-6 py-4">
        <Bone className="h-12 w-full rounded-xl" />
      </div>
    </div>
  );
}

