import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import GrapeDescriptions from "@/components/cellar/GrapeDescriptions";
import { generateGrapeDescription } from "@/lib/api";
import type { DescriptionResponse } from "@/lib/types";

const { refresh } = vi.hoisted(() => ({ refresh: vi.fn() }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ refresh }) }));
vi.mock("@/lib/api", () => ({ generateGrapeDescription: vi.fn() }));
const grapes = [
  { id: 1, name: "cabernet franc", description: "Stored grape description." },
  { id: 2, name: "syrah", description: null },
];
const generated: DescriptionResponse = {
  success: true, description: "New shared grape description.", drink_from_year: null, drink_to_year: null,
};
beforeEach(() => vi.clearAllMocks());

describe("GrapeDescriptions", () => {
  it("renders every grape and cached description without generating on load", () => {
    render(<GrapeDescriptions grapes={grapes} />);
    expect(screen.getByRole("region", { name: "About the Grapes" })).toBeInTheDocument();
    expect(screen.getByText("cabernet franc")).toBeInTheDocument();
    expect(screen.getByText("syrah")).toBeInTheDocument();
    expect(screen.getByText("Stored grape description.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Regenerate cabernet franc description" })).toBeEnabled();
    expect(generateGrapeDescription).not.toHaveBeenCalled();
  });
  it("hides the section when no explicit grapes are linked", () => {
    const { container } = render(<GrapeDescriptions grapes={[]} />);
    expect(container).toBeEmptyDOMElement();
  });
  it("protects a pending request, displays the saved result and refreshes", async () => {
    let resolve!: (result: DescriptionResponse) => void;
    vi.mocked(generateGrapeDescription).mockReturnValue(new Promise((done) => { resolve = done; }));
    render(<GrapeDescriptions grapes={grapes} />);
    const button = screen.getByRole("button", { name: "Generate syrah description" });
    fireEvent.click(button);
    fireEvent.click(button);
    expect(button).toBeDisabled();
    expect(generateGrapeDescription).toHaveBeenCalledExactlyOnceWith(2);
    await act(async () => resolve(generated));
    expect(screen.getByText(generated.description!)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Regenerate syrah description" })).toBeEnabled();
    expect(refresh).toHaveBeenCalledOnce();
  });
  it("retains the previous description after failure and allows retry", async () => {
    vi.mocked(generateGrapeDescription).mockRejectedValueOnce(new Error("Unavailable")).mockResolvedValueOnce(generated);
    render(<GrapeDescriptions grapes={grapes} />);
    const button = screen.getByRole("button", { name: "Regenerate cabernet franc description" });
    fireEvent.click(button);
    expect(await screen.findByRole("alert")).toHaveTextContent("Please retry");
    expect(screen.getByText("Stored grape description.")).toBeInTheDocument();
    expect(refresh).not.toHaveBeenCalled();
    fireEvent.click(button);
    await waitFor(() => expect(screen.getByText(generated.description!)).toBeInTheDocument());
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(refresh).toHaveBeenCalledOnce();
  });
  it("treats an unsuccessful response as a retryable error", async () => {
    vi.mocked(generateGrapeDescription).mockResolvedValue({ ...generated, success: false, description: null });
    render(<GrapeDescriptions grapes={grapes} />);
    fireEvent.click(screen.getByRole("button", { name: "Generate syrah description" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Please retry");
    expect(refresh).not.toHaveBeenCalled();
  });
  it("shows newly refreshed server descriptions", () => {
    const { rerender } = render(<GrapeDescriptions grapes={grapes} />);
    rerender(<GrapeDescriptions grapes={[{ ...grapes[0], description: "Updated on another wine." }]} />);
    expect(screen.getByText("Updated on another wine.")).toBeInTheDocument();
    expect(screen.queryByText("Stored grape description.")).not.toBeInTheDocument();
  });
});
