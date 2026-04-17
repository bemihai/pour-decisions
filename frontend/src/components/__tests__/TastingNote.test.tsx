import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import TastingNote from "@/components/TastingNote";

describe("TastingNote", () => {
  it("strips leading date prefix from a single note", () => {
    render(<TastingNote notes="[2026-03-18] Easy to drink, balanced, with fruits and nice acidity." />);

    expect(screen.getByText("Easy to drink, balanced, with fruits and nice acidity.")).toBeInTheDocument();
    expect(screen.queryByText(/\[2026-03-18]/)).not.toBeInTheDocument();
  });

  it("renders multiple notes as list items", () => {
    render(
      <TastingNote
        notes={"[2026-03-18] Fresh and balanced\n[2026-03-25] More floral on day two"}
      />,
    );

    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(2);
    expect(items[0]).toHaveTextContent("Fresh and balanced");
    expect(items[1]).toHaveTextContent("More floral on day two");
    expect(screen.queryByText(/\[2026-03-18]/)).not.toBeInTheDocument();
    expect(screen.queryByText(/\[2026-03-25]/)).not.toBeInTheDocument();
  });

  it("deduplicates repeated note entries", () => {
    render(
      <TastingNote
        notes={"[2026-03-18] Fresh and balanced\n[2026-03-18] Fresh and balanced\n[2026-03-25] More floral"}
      />,
    );

    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(2);
    expect(items[0]).toHaveTextContent("Fresh and balanced");
    expect(items[1]).toHaveTextContent("More floral");
  });
});

