import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  createTastePreference,
  deleteChatThread,
  deleteTastePreference,
  getTastePreferenceOptions,
  getTastePreferences,
  resetTastePreferences,
  updateTastePreference,
} from "@/lib/api";


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

describe("taste preference client", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("uses the approved list and options routes", async () => {
    const fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({ items: [], total: 0, max_items: 100 }),
    });
    vi.stubGlobal("fetch", fetch);

    await getTastePreferences();
    await getTastePreferenceOptions();

    expect(fetch).toHaveBeenNthCalledWith(1, "http://localhost:8000/api/taste-profile/preferences", {
      headers: { "Content-Type": "application/json" },
    });
    expect(fetch).toHaveBeenNthCalledWith(2, "http://localhost:8000/api/taste-profile/preferences/options", {
      headers: { "Content-Type": "application/json" },
    });
  });

  it("sends exact create, update, delete, and reset contracts", async () => {
    const fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: vi.fn().mockResolvedValue({ id: 7 }),
    });
    vi.stubGlobal("fetch", fetch);

    await createTastePreference({ subject_kind: "grape", stance: "like", value: "Riesling" });
    await updateTastePreference(7, { expected_version: 2, stance: "avoid" });
    await deleteTastePreference(7, 3);
    await resetTastePreferences({ confirm: true, expected_count: 4 });

    expect(fetch).toHaveBeenNthCalledWith(
      1,
      "http://localhost:8000/api/taste-profile/preferences",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ subject_kind: "grape", stance: "like", value: "Riesling" }),
      }),
    );
    expect(fetch).toHaveBeenNthCalledWith(
      2,
      "http://localhost:8000/api/taste-profile/preferences/7",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({ expected_version: 2, stance: "avoid" }),
      }),
    );
    expect(fetch).toHaveBeenNthCalledWith(
      3,
      "http://localhost:8000/api/taste-profile/preferences/7?expected_version=3",
      expect.objectContaining({ method: "DELETE" }),
    );
    expect(fetch).toHaveBeenNthCalledWith(
      4,
      "http://localhost:8000/api/taste-profile/preferences/reset",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ confirm: true, expected_count: 4 }),
      }),
    );
  });

  it("preserves bounded object detail in ApiError", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 409,
        json: vi.fn().mockResolvedValue({
          detail: { code: "preference_version_conflict", message: "Preference changed." },
        }),
      }),
    );

    await expect(getTastePreferences()).rejects.toEqual(
      new ApiError(409, "Preference changed.", "preference_version_conflict"),
    );
  });
});
