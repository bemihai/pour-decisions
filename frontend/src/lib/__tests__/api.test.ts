import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, deleteChatThread } from "@/lib/api";


describe("deleteChatThread", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("accepts an idempotent 204 response without parsing a body", async () => {
    const json = vi.fn();
    const fetch = vi.fn().mockResolvedValue({ ok: true, status: 204, json });
    vi.stubGlobal("fetch", fetch);

    await expect(deleteChatThread("thread/id")).resolves.toBeUndefined();

    expect(fetch).toHaveBeenCalledWith(
      "http://localhost:8000/api/chat/threads/thread%2Fid",
      { method: "DELETE" },
    );
    expect(json).not.toHaveBeenCalled();
  });

  it("raises a typed API error with the server detail", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 500,
        json: vi.fn().mockResolvedValue({ detail: "Deletion failed" }),
      }),
    );

    await expect(deleteChatThread("thread-id")).rejects.toEqual(
      new ApiError(500, "Deletion failed"),
    );
  });
});
