import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import TastePreferences from "@/components/taste-profile/TastePreferences";
import {
  ApiError,
  createTastePreference,
  deleteTastePreference,
  getTastePreferenceOptions,
  getTastePreferences,
  resetTastePreferences,
  updateTastePreference,
} from "@/lib/api";
import type {
  PreferenceListResponse,
  PreferenceOptionsResponse,
  PreferenceResponse,
} from "@/lib/types";

vi.mock("sonner", () => ({ toast: { success: vi.fn() } }));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    createTastePreference: vi.fn(),
    deleteTastePreference: vi.fn(),
    getTastePreferenceOptions: vi.fn(),
    getTastePreferences: vi.fn(),
    resetTastePreferences: vi.fn(),
    updateTastePreference: vi.fn(),
  };
});

const preference: PreferenceResponse = {
  id: 7,
  subject_kind: "grape",
  stance: "like",
  normalized_value: "riesling",
  display_value: "Riesling",
  price_amount: null,
  currency: null,
  provenance: "explicit_user",
  version: 2,
  created_at: "2026-10-05T10:00:00Z",
  updated_at: "2026-10-05T10:00:00Z",
};

const options: PreferenceOptionsResponse = {
  subject_kinds: ["grape", "region", "producer", "wine_style", "price_ceiling"],
  stances: ["like", "dislike", "avoid"],
  grapes: ["Riesling", "Syrah"],
  regions: ["Mosel"],
  producers: ["Example Estate"],
  wine_styles: ["red", "white"],
  currencies: ["EUR", "RON"],
  max_items: 100,
};

function list(items: PreferenceResponse[] = []): PreferenceListResponse {
  return { items, total: items.length, max_items: 100 };
}

function renderPreferences() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  return { ...render(<TastePreferences />, { wrapper }), queryClient };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

describe("TastePreferences", () => {
  beforeEach(() => {
    vi.mocked(getTastePreferences).mockResolvedValue(list());
    vi.mocked(getTastePreferenceOptions).mockResolvedValue(options);
    vi.mocked(createTastePreference).mockResolvedValue(preference);
    vi.mocked(updateTastePreference).mockResolvedValue(preference);
    vi.mocked(deleteTastePreference).mockResolvedValue({ id: 7, deleted_version: 2 });
    vi.mocked(resetTastePreferences).mockResolvedValue({ deleted_count: 1 });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("announces an accessible loading skeleton", () => {
    vi.mocked(getTastePreferences).mockReturnValue(new Promise(() => undefined));
    renderPreferences();
    expect(screen.getByRole("status")).toHaveTextContent("Loading preferences");
  });

  it("shows a bounded list error and retries on request", async () => {
    vi.mocked(getTastePreferences)
      .mockRejectedValueOnce(new ApiError(503, "Preference store is busy.", "preference_store_busy"))
      .mockResolvedValueOnce(list());
    const user = userEvent.setup();
    renderPreferences();

    expect(await screen.findByRole("alert")).toHaveTextContent("Preference store is busy.");
    await user.click(screen.getByRole("button", { name: "Retry" }));

    expect(await screen.findByRole("heading", { name: "Declared preferences" })).toBeInTheDocument();
    expect(getTastePreferences).toHaveBeenCalledTimes(2);
  });

  it("keeps an empty list usable when options fail and disables Add", async () => {
    vi.mocked(getTastePreferenceOptions).mockRejectedValue(new ApiError(500, "Options unavailable."));
    renderPreferences();

    expect(await screen.findByText("No declared preferences yet")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add preference" })).toBeDisabled();
    expect(screen.getByRole("alert")).toHaveTextContent("Add is disabled");
  });

  it("renders responsive cards without exposing normalized identity or timestamps", async () => {
    vi.mocked(getTastePreferences).mockResolvedValue(list([preference]));
    const { container } = renderPreferences();

    expect(await screen.findByText("Riesling")).toBeInTheDocument();
    expect(screen.getByText("Explicitly added by you")).toBeInTheDocument();
    expect(screen.queryByText("riesling")).not.toBeInTheDocument();
    expect(screen.queryByText(preference.created_at)).not.toBeInTheDocument();
    expect(container.querySelector("section.min-w-0")).toBeInTheDocument();
    expect(container.querySelector(".sm\\:grid-cols-2")).toBeInTheDocument();
  });

  it("keeps add controls disabled while pending, then restores focus after success", async () => {
    const createResult = deferred<PreferenceResponse>();
    vi.mocked(createTastePreference).mockReturnValue(createResult.promise);
    const user = userEvent.setup();
    renderPreferences();

    const trigger = await screen.findByRole("button", { name: "Add preference" });
    await user.click(trigger);
    const dialog = screen.getByRole("dialog", { name: "Add preference" });
    await user.click(within(dialog).getByRole("button", { name: "Add preference" }));

    expect(createTastePreference).toHaveBeenCalledWith({
      subject_kind: "grape",
      stance: "like",
      value: "Riesling",
    });
    expect(within(dialog).getByRole("button", { name: "Add preference" })).toBeDisabled();
    expect(within(dialog).getByRole("button", { name: "Cancel" })).toBeDisabled();
    expect(within(dialog).getByText("Saving preference…")).toBeInTheDocument();

    createResult.resolve(preference);
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(trigger).toHaveFocus();
  });

  it("preserves add values and exposes a service failure beside the form", async () => {
    vi.mocked(createTastePreference).mockRejectedValue(
      new ApiError(422, "Preference value could not be resolved.", "preference_value_unresolved"),
    );
    const user = userEvent.setup();
    renderPreferences();

    await user.click(await screen.findByRole("button", { name: "Add preference" }));
    const dialog = screen.getByRole("dialog", { name: "Add preference" });
    await user.selectOptions(within(dialog).getByLabelText("Value"), "Syrah");
    await user.click(within(dialog).getByRole("button", { name: "Add preference" }));

    expect(await within(dialog).findByText("Preference value could not be resolved.")).toBeInTheDocument();
    expect(within(dialog).getByLabelText("Value")).toHaveValue("Syrah");
  });

  it("edits only mutable fields and refreshes an actionable conflict", async () => {
    vi.mocked(getTastePreferences).mockResolvedValue(list([preference]));
    vi.mocked(updateTastePreference).mockRejectedValue(
      new ApiError(409, "Preference changed.", "preference_version_conflict"),
    );
    const user = userEvent.setup();
    renderPreferences();

    await user.click(await screen.findByRole("button", { name: "Edit Riesling" }));
    const dialog = screen.getByRole("dialog", { name: "Edit Riesling" });
    await user.selectOptions(within(dialog).getByLabelText("Stance"), "avoid");
    await user.click(within(dialog).getByRole("button", { name: "Save changes" }));

    expect(updateTastePreference).toHaveBeenCalledWith(7, { expected_version: 2, stance: "avoid" });
    expect(await within(dialog).findByText(/list has been refreshed/)).toBeInTheDocument();
    expect(getTastePreferences).toHaveBeenCalledTimes(2);
    expect(within(dialog).getByLabelText("Stance")).toHaveValue("avoid");
  });

  it("confirms an edit before closing and restoring focus", async () => {
    vi.mocked(getTastePreferences).mockResolvedValue(list([preference]));
    const user = userEvent.setup();
    renderPreferences();

    const trigger = await screen.findByRole("button", { name: "Edit Riesling" });
    await user.click(trigger);
    const dialog = screen.getByRole("dialog", { name: "Edit Riesling" });
    await user.selectOptions(within(dialog).getByLabelText("Stance"), "dislike");
    await user.click(within(dialog).getByRole("button", { name: "Save changes" }));

    await waitFor(() =>
      expect(updateTastePreference).toHaveBeenCalledWith(7, { expected_version: 2, stance: "dislike" }),
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    await waitFor(() => expect(trigger).toHaveFocus());
  });

  it("names a delete confirmation and keeps failures visible", async () => {
    vi.mocked(getTastePreferences).mockResolvedValue(list([preference]));
    vi.mocked(deleteTastePreference).mockRejectedValue(new ApiError(500, "Delete unavailable."));
    const user = userEvent.setup();
    renderPreferences();

    await user.click(await screen.findByRole("button", { name: "Delete Riesling" }));
    const dialog = screen.getByRole("dialog", { name: "Delete preference" });
    expect(dialog).toHaveTextContent("Delete Riesling?");
    await user.click(within(dialog).getByRole("button", { name: "Delete preference" }));

    expect(deleteTastePreference).toHaveBeenCalledWith(7, 2);
    expect(await within(dialog).findByText("Delete unavailable.")).toBeInTheDocument();
  });

  it("closes a delete dialog only after server confirmation", async () => {
    vi.mocked(getTastePreferences).mockResolvedValue(list([preference]));
    const user = userEvent.setup();
    renderPreferences();

    const trigger = await screen.findByRole("button", { name: "Delete Riesling" });
    await user.click(trigger);
    const dialog = screen.getByRole("dialog", { name: "Delete preference" });
    await user.click(within(dialog).getByRole("button", { name: "Delete preference" }));

    await waitFor(() => expect(deleteTastePreference).toHaveBeenCalledWith(7, 2));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    await waitFor(() => expect(trigger).toHaveFocus());
  });

  it("requires typed RESET and sends the displayed expected count", async () => {
    vi.mocked(getTastePreferences).mockResolvedValue(list([preference]));
    const user = userEvent.setup();
    renderPreferences();

    const trigger = await screen.findByRole("button", { name: "Reset all preferences" });
    await user.click(trigger);
    const dialog = screen.getByRole("dialog", { name: "Reset all preferences" });
    const confirm = within(dialog).getByRole("button", { name: "Reset all preferences" });
    expect(confirm).toBeDisabled();

    await user.type(within(dialog).getByLabelText("Type RESET"), "RESET");
    expect(confirm).toBeEnabled();
    await user.click(confirm);

    await waitFor(() => expect(resetTastePreferences).toHaveBeenCalledWith({ confirm: true, expected_count: 1 }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(trigger).toHaveFocus();
  });

  it("preserves reset confirmation after a service failure", async () => {
    vi.mocked(getTastePreferences).mockResolvedValue(list([preference]));
    vi.mocked(resetTastePreferences).mockRejectedValue(new ApiError(500, "Reset unavailable."));
    const user = userEvent.setup();
    renderPreferences();

    await user.click(await screen.findByRole("button", { name: "Reset all preferences" }));
    const dialog = screen.getByRole("dialog", { name: "Reset all preferences" });
    const input = within(dialog).getByLabelText("Type RESET");
    await user.type(input, "RESET");
    await user.click(within(dialog).getByRole("button", { name: "Reset all preferences" }));

    expect(await within(dialog).findByText("Reset unavailable.")).toBeInTheDocument();
    expect(input).toHaveValue("RESET");
  });

  it("supports keyboard dismissal and restores focus to the initiating control", async () => {
    const user = userEvent.setup();
    renderPreferences();
    const trigger = await screen.findByRole("button", { name: "Add preference" });

    trigger.focus();
    await user.keyboard("{Enter}");
    expect(screen.getByRole("dialog", { name: "Add preference" })).toBeInTheDocument();
    await user.keyboard("{Escape}");

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(trigger).toHaveFocus();
  });
});
