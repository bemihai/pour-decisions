import { describe, expect, it } from "vitest";
import { CHAT_COMMANDS, expandChatHistoryContent, resolveChatCommand } from "@/lib/chat-commands";

describe("chat commands", () => {
  it("matches case and surrounding whitespace while preserving argument text", () => {
    const result = resolveChatCommand("  /PAIR   grilled  salmon \n", "intelligent");
    expect(result.kind).toBe("request");
    if (result.kind !== "request") return;
    expect(result.message).toContain("grilled  salmon");
    expect(result.message).toContain("Only recommend owned wines");
    expect(result.message).toContain("if no suitable owned wine is found, say so");
  });
  it.each(["/pair", "/price", "/describe", "/unknown", "/help extra", "/pair <food>", "/price <wine> [vintage]"])("rejects invalid input %s", (text) => {
    expect(resolveChatCommand(text, "intelligent").kind).toBe("guidance");
  });
  it("requests sourced current prices in original currencies", () => {
    const result = resolveChatCommand("/price Château Margaux 2000", "intelligent");
    expect(result).toEqual({ kind: "request", message: expect.stringContaining("Château Margaux 2000") });
    expect(JSON.stringify(result)).toContain("current online retail listings");
    expect(JSON.stringify(result)).toContain("without converting currencies");
    expect(JSON.stringify(result)).toContain("source links");
  });
  it("requests grounded descriptions without database persistence", () => {
    const result = resolveChatCommand("/describe Château Margaux 2000", "intelligent");
    expect(JSON.stringify(result)).toContain("comprehensive evidence-grounded");
    expect(JSON.stringify(result)).toContain("do not generate, update, or persist a database description");
  });
  it("keeps help local with usage and examples for all four commands", () => {
    expect(resolveChatCommand("/HELP", "intelligent")).toEqual({ kind: "help" });
    expect(CHAT_COMMANDS).toHaveLength(4);
    for (const command of CHAT_COMMANDS) {
      expect(command.purpose).toBeTruthy();
      expect(command.example.startsWith(command.name)).toBe(true);
    }
  });
  it("blocks slash commands in RAG mode and preserves free-form text", () => {
    expect(resolveChatCommand("/pair salmon", "rag_only").kind).toBe("guidance");
    expect(resolveChatCommand("Wine / food?", "rag_only")).toEqual({ kind: "request", message: "Wine / food?" });
  });
  it("expands history consistently and preserves ordinary and invalid historical text", () => {
    expect(expandChatHistoryContent("/pair salmon", "intelligent")).toBe(
      (resolveChatCommand("/pair salmon", "intelligent") as { message: string }).message,
    );
    for (const text of ["Normal text", "/help", "/unknown"]) {
      expect(expandChatHistoryContent(text, "intelligent")).toBe(text);
    }
  });
});
