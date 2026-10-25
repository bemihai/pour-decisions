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
  it.each([
    "/pair",
    "/price",
    "/describe",
    "/unknown",
    "/help extra",
    "/profile extra",
    "/pair <food>",
    "/price <wine> [vintage]",
    "/price Château Margaux [vintage]",
    "/cellar [query]",
    "/open [occasion or meal]",
    "/compare",
    "/compare one wine",
    "/compare one wine ;",
    "/compare one wine ;; another wine",
    "/compare <wine A> ; <wine B> [; …]",
    "/compare Barolo ; Brunello [; …]",
  ])("rejects invalid input %s", (text) => {
    expect(resolveChatCommand(text, "intelligent").kind).toBe("guidance");
  });
  it.each([
    ["/cellar", "Summarize the wines currently owned", "complete totals"],
    ["/cellar Romanian  reds", "Romanian  reds", "matching wine names"],
    ["/open", "Recommend one wine currently owned", "Do not recommend an unowned wine"],
    ["/open roast  lamb tonight", "roast  lamb tonight", "Ask a focused clarification"],
    ["/history", "recent recorded tasting history", "Include unrated records"],
    ["/history Crama  Bauer", "Crama  Bauer", "rated and unrated records"],
    ["/profile", "current wine taste profile", "Do not create or modify preferences"],
  ])("expands %s with approved request semantics", (text, expected, policy) => {
    const result = resolveChatCommand(text, "intelligent");
    expect(result).toEqual({ kind: "request", message: expect.stringContaining(expected) });
    expect(JSON.stringify(result)).toContain(policy);
  });
  it("preserves two or more comparison identities on separate lines", () => {
    const result = resolveChatCommand(
      " /COMPARE  Povero  Barolo 2019 ; Avincis Alutus Primus 2019 ; Château Margaux 2000 ",
      "intelligent",
    );
    expect(result).toEqual({
      kind: "request",
      message: expect.stringContaining(
        "Povero  Barolo 2019\nAvincis Alutus Primus 2019\nChâteau Margaux 2000",
      ),
    });
    expect(JSON.stringify(result)).toContain("The wines need not be owned");
    expect(JSON.stringify(result)).toContain("Do not infer ownership");
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
  it("keeps help local with usage and examples for all commands", () => {
    expect(resolveChatCommand("/HELP", "intelligent")).toEqual({ kind: "help" });
    expect(CHAT_COMMANDS).toHaveLength(9);
    for (const command of CHAT_COMMANDS) {
      expect(command.purpose).toBeTruthy();
      expect(command.example.startsWith(command.name)).toBe(true);
    }
  });
  it("blocks slash commands in RAG mode and preserves free-form text", () => {
    expect(resolveChatCommand("/pair salmon", "rag_only").kind).toBe("guidance");
    expect(resolveChatCommand("/profile", "rag_only").kind).toBe("guidance");
    expect(resolveChatCommand("Wine / food?", "rag_only")).toEqual({ kind: "request", message: "Wine / food?" });
  });
  it("expands history consistently and preserves ordinary and invalid historical text", () => {
    for (const text of ["/pair salmon", "/cellar", "/compare Barolo ; Brunello", "/profile"]) {
      expect(expandChatHistoryContent(text, "intelligent")).toBe(
        (resolveChatCommand(text, "intelligent") as { message: string }).message,
      );
    }
    for (const text of ["Normal text", "/help", "/unknown"]) {
      expect(expandChatHistoryContent(text, "intelligent")).toBe(text);
    }
  });
});
