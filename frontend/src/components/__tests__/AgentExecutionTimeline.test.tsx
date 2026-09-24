import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import AgentExecutionTimeline from "@/components/AgentExecutionTimeline";


describe("AgentExecutionTimeline", () => {
  it("renders allowlisted labels and bounded statuses accessibly", () => {
    render(
      <AgentExecutionTimeline
        events={[
          {
            type: "tool_progress",
            invocation_id: 1,
            tool_key: "search_wine_knowledge",
            status: "completed",
          },
          {
            type: "tool_progress",
            invocation_id: 2,
            tool_key: "unknown-server-key",
            status: "failed",
          },
        ]}
      />,
    );

    expect(screen.getByRole("status", { name: "Agent progress" })).toBeInTheDocument();
    expect(screen.getByText("Searching wine knowledge")).toBeInTheDocument();
    expect(screen.getByText("Using a wine tool")).toBeInTheDocument();
    expect(screen.getByText("Completed")).toBeInTheDocument();
    expect(screen.getByText("Could not complete")).toBeInTheDocument();
  });

  it("renders nothing before progress arrives", () => {
    const { container } = render(<AgentExecutionTimeline events={[]} />);
    expect(container).toBeEmptyDOMElement();
  });
});
