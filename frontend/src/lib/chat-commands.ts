import type { AgentMode } from "@/lib/types";

export const CHAT_COMMANDS = [
  { name: "/pair", hint: "<food>", purpose: "Pair food with wines in your cellar", example: "/pair grilled salmon" },
  { name: "/price", hint: "<wine> [vintage]", purpose: "Find current online retail prices", example: "/price Château Margaux 2000" },
  { name: "/describe", hint: "<wine> [vintage]", purpose: "Explain a wine using verified evidence", example: "/describe Château Margaux 2000" },
  { name: "/help", hint: "", purpose: "Show command usage and examples", example: "/help" },
] as const;

export type ChatCommand = (typeof CHAT_COMMANDS)[number];
export type CommandResolution =
  | { kind: "request"; message: string }
  | { kind: "guidance"; message: string }
  | { kind: "help" };

/** Resolve only leading slash commands; keep the user's argument text intact. */
export function resolveChatCommand(text: string, mode: AgentMode): CommandResolution {
  const trimmed = text.trim();
  if (!trimmed.startsWith("/")) return { kind: "request", message: text };
  if (mode !== "intelligent") {
    return { kind: "guidance", message: "Slash commands require Intelligent mode. Select Intelligent mode to use them." };
  }
  const [name] = trimmed.split(/\s+/, 1);
  const command = CHAT_COMMANDS.find((entry) => entry.name === name.toLowerCase());
  if (!command) return { kind: "guidance", message: "Unknown command. Use /help to see available commands." };
  const argument = trimmed.slice(name.length).trim();
  if (command.name === "/help") {
    return argument ? { kind: "guidance", message: "Usage: /help" } : { kind: "help" };
  }
  if (!argument || argument.includes("<food>") || argument.includes("<wine>")) {
    return { kind: "guidance", message: `Usage: ${command.name} ${command.hint}` };
  }
  switch (command.name) {
    case "/pair":
      return { kind: "request", message: `Recommend suitable wines currently present in my cellar to pair with this food: ${argument}. Verify current ownership using my cellar data and explain the matches using verified evidence. Only recommend owned wines; if no suitable owned wine is found, say so rather than recommending an unowned bottle.` };
    case "/price":
      return { kind: "request", message: `Find and summarize current online retail listings for this wine and vintage (if specified): ${argument}. Preserve each listing's stated currency without converting currencies, and include source links. State clearly when current prices cannot be verified.` };
    case "/describe":
      return { kind: "request", message: `Provide a comprehensive evidence-grounded description of this wine and vintage (if specified): ${argument}. Use verified evidence and disclose sources and gaps following your grounding policy. Answer only in this conversation; do not generate, update, or persist a database description.` };
  }
}

/** Expand valid historical human commands without rewriting stored messages. */
export function expandChatHistoryContent(text: string, mode: AgentMode): string {
  const result = resolveChatCommand(text, mode);
  return result.kind === "request" ? result.message : text;
}
