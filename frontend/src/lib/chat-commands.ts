import type { AgentMode } from "@/lib/types";

export const CHAT_COMMANDS = [
  { name: "/pair", hint: "<food>", purpose: "Pair food with wines in your cellar", example: "/pair grilled salmon" },
  { name: "/cellar", hint: "[query]", purpose: "Explore wines currently in your cellar", example: "/cellar Romanian reds" },
  { name: "/open", hint: "[occasion or meal]", purpose: "Choose an owned bottle to open now", example: "/open roast lamb tonight" },
  {
    name: "/compare",
    hint: "<wine A> ; <wine B> [; …]",
    purpose: "Compare two or more owned or unowned wines",
    example: "/compare Povero Barolo 2019 ; Avincis Alutus Primus 2019",
  },
  { name: "/history", hint: "[query]", purpose: "Review your recorded tasting history", example: "/history Crama Bauer" },
  { name: "/profile", hint: "", purpose: "Summarize your current wine taste profile", example: "/profile" },
  { name: "/price", hint: "<wine> [vintage]", purpose: "Find current online retail prices", example: "/price Château Margaux 2000" },
  { name: "/describe", hint: "<wine> [vintage]", purpose: "Explain a wine using verified evidence", example: "/describe Château Margaux 2000" },
  { name: "/help", hint: "", purpose: "Show command usage and examples", example: "/help" },
] as const;

export type ChatCommand = (typeof CHAT_COMMANDS)[number];
export type CommandResolution =
  | { kind: "request"; message: string }
  | { kind: "guidance"; message: string }
  | { kind: "help" };

const ARGUMENT_PLACEHOLDER_PATTERN = /<[^>]+>|\[(?:query|occasion or meal|vintage|; …)\]/iu;

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
  if (command.name === "/profile") {
    return argument
      ? { kind: "guidance", message: "Usage: /profile" }
      : {
          kind: "request",
          message: "Summarize my current wine taste profile using verified profile data. Clearly distinguish explicitly declared preferences from patterns observed in rated tasting history. Mention relevant conflicts and limited evidence without inventing preferences. If either source is empty, state that limitation. Do not create or modify preferences.",
        };
  }
  if (ARGUMENT_PLACEHOLDER_PATTERN.test(argument)) {
    return { kind: "guidance", message: `Usage: ${command.name} ${command.hint}` };
  }
  switch (command.name) {
    case "/pair":
      if (!argument) return { kind: "guidance", message: `Usage: ${command.name} ${command.hint}` };
      return { kind: "request", message: `Recommend suitable wines currently present in my cellar to pair with this food: ${argument}. Verify current ownership using my cellar data and explain the matches using verified evidence. Only recommend owned wines; if no suitable owned wine is found, say so rather than recommending an unowned bottle.` };
    case "/cellar":
      return argument
        ? { kind: "request", message: `Search wines currently owned in my cellar for this request: ${argument}. Use verified cellar data and report matching wine names, vintages, owned quantities, and drinking status when available. Do not present a limited result set as the complete cellar. If no wines match, say so.` }
        : { kind: "request", message: "Summarize the wines currently owned in my cellar using verified cellar data. Include bottle and wine totals and a concise breakdown by wine type and country when available. Distinguish complete totals from limited search results. If the cellar is empty, say so." };
    case "/open": {
      const recommendation = "Recommend one wine currently owned in my cellar to open now. Verify ownership and consider its recorded drinking window and my declared preferences and observed taste profile. Explain the choice briefly, distinguishing supported facts from missing information. Do not recommend an unowned wine or record any bottle as consumed. If no suitable owned wine can be identified, say so.";
      return {
        kind: "request",
        message: argument
          ? `${recommendation} Consider this occasion, meal, or constraint: ${argument}. Ask a focused clarification if a material ambiguity prevents a suitable recommendation.`
          : recommendation,
      };
    }
    case "/compare": {
      const wines = argument.split(";").map((wine) => wine.trim());
      if (wines.length < 2 || wines.some((wine) => !wine)) {
        return { kind: "guidance", message: `Usage: ${command.name} ${command.hint}` };
      }
      return {
        kind: "request",
        message: `Compare these wines using verified evidence:\n${wines.join("\n")}\nThe wines need not be owned. Compare their identity, origin, grapes, documented style, and drinking windows where supported, noting meaningful similarities and differences. Distinguish vintage-specific evidence from general wine information. Do not infer ownership or invent missing facts. Ask for clarification when an ambiguous identity materially affects the comparison. Follow the existing grounding and source-disclosure policy.`,
      };
    }
    case "/history":
      return argument
        ? { kind: "request", message: `Find my recorded tasting history matching this request: ${argument}. Use tasting-history data, include rated and unrated records, and present the most recent matches first with dates, ratings, and notes when available. Distinguish returned records from the total matching history. Do not treat missing ratings as dislike or invent past tastings. If there are no matches, say so.` }
        : { kind: "request", message: "Summarize my recent recorded tasting history, newest first, using tasting-history data. Include wine names, vintages, consumption dates, personal ratings, and notes when available. Include unrated records. Distinguish returned records from the total matching history and do not treat missing ratings as dislike. If there are no records, say so." };
    case "/price":
      if (!argument) return { kind: "guidance", message: `Usage: ${command.name} ${command.hint}` };
      return { kind: "request", message: `Find and summarize current online retail listings for this wine and vintage (if specified): ${argument}. Preserve each listing's stated currency without converting currencies, and include source links. State clearly when current prices cannot be verified.` };
    case "/describe":
      if (!argument) return { kind: "guidance", message: `Usage: ${command.name} ${command.hint}` };
      return { kind: "request", message: `Provide a comprehensive evidence-grounded description of this wine and vintage (if specified): ${argument}. Use verified evidence and disclose sources and gaps following your grounding policy. Answer only in this conversation; do not generate, update, or persist a database description.` };
  }
}

/** Expand valid historical human commands without rewriting stored messages. */
export function expandChatHistoryContent(text: string, mode: AgentMode): string {
  const result = resolveChatCommand(text, mode);
  return result.kind === "request" ? result.message : text;
}
