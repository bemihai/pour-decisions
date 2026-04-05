import ApiHealthCheck from "@/components/ApiHealthCheck";

export default function ChatbotPage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4 px-4">
      <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-800 via-purple-600 to-purple-700 bg-clip-text text-transparent">
        Pour Decisions
      </h1>
      <p className="text-xl text-brand-green font-medium">Let the bot choose your bottle</p>
      <p className="text-muted-foreground">Chatbot page coming soon.</p>
      <ApiHealthCheck />
    </div>
  );
}
