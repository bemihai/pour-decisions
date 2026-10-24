import { afterEach, expect, it, vi } from "vitest";
import { generateGrapeDescription } from "@/lib/api";
afterEach(() => vi.unstubAllGlobals());
it("posts the established RAG and web options to the grape endpoint", async () => {
  const fetch = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ success: true, description: "Saved" }) });
  vi.stubGlobal("fetch", fetch);
  await generateGrapeDescription(42);
  expect(fetch).toHaveBeenCalledWith("http://localhost:8000/api/grapes/42/description", expect.objectContaining({
    method: "POST", body: JSON.stringify({ use_rag_context: true, use_web_search: true }),
  }));
});
