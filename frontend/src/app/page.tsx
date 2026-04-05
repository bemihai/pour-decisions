import PageHeader from "@/components/PageHeader";
import ChatInterface from "@/components/ChatInterface";
import ChatSidebar, { MobileSidebarTrigger } from "@/components/ChatSidebar";

/**
 * Chatbot page — root route (/).
 *
 * Composes PageHeader, ChatInterface, and ChatSidebar into a full-height
 * layout that fills the viewport below the navigation bar.
 * Replaces src/ui/pages/chatbot.py.
 */
export default function ChatbotPage() {
  return (
    // h-[calc(100vh-3.5rem)]: fill viewport minus the h-14 nav bar.
    <div className="flex flex-col h-[calc(100vh-3.5rem)]">
      {/* Page header with mobile settings trigger */}
      <header className="shrink-0 flex items-center justify-between px-6 pt-5 pb-4 border-b border-border">
        <PageHeader
          title="Pour Decisions"
          subtitle="Let the bot choose your bottle"
          compact
          className="mb-0"
        />
        <MobileSidebarTrigger />
      </header>

      {/* Chat area + sidebar — min-h-0 lets the flex children shrink so the
          message list can scroll without the row growing past the viewport. */}
      <div className="flex flex-1 min-h-0">
        {/* flex-1 min-w-0: take all remaining width; min-w-0 prevents overflow */}
        <div className="flex-1 min-w-0">
          <ChatInterface />
        </div>
        <ChatSidebar />
      </div>
    </div>
  );
}
