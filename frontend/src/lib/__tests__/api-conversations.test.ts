import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  createConversation,
  deleteConversation,
  getConversation,
  getConversationOperation,
  getConversationTurns,
  listConversations,
  renameConversation,
} from "@/lib/api";

describe("saved conversation client", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("uses the lifecycle, paging, and operation routes with encoded identity", async () => {
    const fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: vi.fn().mockResolvedValue({ items: [] }),
    });
    vi.stubGlobal("fetch", fetch);

    await createConversation({ id: "conversation/id", agent_mode: "rag_only", first_message: "Hi" });
    await listConversations(20, "next cursor");
    await getConversation("conversation/id");
    await getConversationTurns("conversation/id", 40, 12);
    await renameConversation("conversation/id", { title: "Dinner", expected_version: 3 });
    await deleteConversation("conversation/id", 4);
    await getConversationOperation("conversation/id", "operation/id");

    expect(fetch).toHaveBeenNthCalledWith(
      1,
      "http://localhost:8000/api/conversations",
      expect.objectContaining({ method: "POST" }),
    );
    expect(fetch).toHaveBeenNthCalledWith(
      2,
      "http://localhost:8000/api/conversations?limit=20&cursor=next+cursor",
      expect.any(Object),
    );
    expect(fetch).toHaveBeenNthCalledWith(
      4,
      "http://localhost:8000/api/conversations/conversation%2Fid/turns?limit=40&before_sequence=12",
      expect.any(Object),
    );
    expect(fetch).toHaveBeenNthCalledWith(
      6,
      "http://localhost:8000/api/conversations/conversation%2Fid?expected_version=4",
      { method: "DELETE" },
    );
    expect(fetch).toHaveBeenNthCalledWith(
      7,
      "http://localhost:8000/api/conversations/conversation%2Fid/operations/operation%2Fid",
      expect.any(Object),
    );
  });

  it("preserves refreshable conflict metadata", async () => {
    const current = {
      id: "123e4567-e89b-12d3-a456-426614174000",
      agent_mode: "intelligent",
      title: "Current",
      version: 8,
      state: "active",
      created_at: "2026-10-28T00:00:00Z",
      updated_at: "2026-10-28T00:00:00Z",
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 409,
        json: vi.fn().mockResolvedValue({ detail: { code: "stale_version", current } }),
      }),
    );

    await expect(getConversation(current.id)).rejects.toEqual(
      new ApiError(409, "HTTP 409", "stale_version", { current }),
    );
  });
});
