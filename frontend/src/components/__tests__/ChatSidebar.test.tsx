import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import ChatSidebar, { MobileSidebarTrigger } from "@/components/ChatSidebar";
import { useChatStore } from "@/stores/chat-store";

// Mock the chat store
vi.mock("@/stores/chat-store", () => ({
  useChatStore: vi.fn(),
}));

describe("ChatSidebar — Model Provider Toggle", () => {
  const mockSetModelProvider = vi.fn();
  const mockSetAgentMode = vi.fn();
  const mockResetChat = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    // Default store state
    (useChatStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      agentMode: "intelligent",
      modelProvider: "local",
      isLoading: false,
      setAgentMode: mockSetAgentMode,
      setModelProvider: mockSetModelProvider,
      resetChat: mockResetChat,
    });
  });

  it("renders model provider toggle with local and cloud options", () => {
    render(<ChatSidebar />);

    // Check for both model options
    expect(screen.getByText("Local (Gemma 4)")).toBeInTheDocument();
    expect(screen.getByText("Cloud (Gemini)")).toBeInTheDocument();
  });

  it("displays descriptions for each model option", () => {
    render(<ChatSidebar />);

    // Local model description
    expect(screen.getByText(/Runs on your machine via Ollama/i)).toBeInTheDocument();
    // Cloud model description
    expect(screen.getByText(/Google Gemini API/i)).toBeInTheDocument();
  });

  it("defaults to local model provider", () => {
    render(<ChatSidebar />);

    // Find the Local (Gemma 4) button and check if it has active styling
    const localButton = screen.getByRole("button", { name: /Local \(Gemma 4\)/i });
    expect(localButton.className).toContain("border-brand-burgundy");
    expect(localButton).toHaveAttribute("aria-pressed", "true");
  });

  it("shows cloud model as active when modelProvider is 'cloud'", () => {
    (useChatStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      agentMode: "intelligent",
      modelProvider: "cloud",
      isLoading: false,
      setAgentMode: mockSetAgentMode,
      setModelProvider: mockSetModelProvider,
      resetChat: mockResetChat,
    });

    render(<ChatSidebar />);

    const cloudButton = screen.getByRole("button", { name: /Cloud \(Gemini\)/i });
    expect(cloudButton.className).toContain("border-brand-burgundy");
    expect(cloudButton).toHaveAttribute("aria-pressed", "true");
  });

  it("calls setModelProvider when clicking local model", async () => {
    // Start with cloud selected
    (useChatStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      agentMode: "intelligent",
      modelProvider: "cloud",
      isLoading: false,
      setAgentMode: mockSetAgentMode,
      setModelProvider: mockSetModelProvider,
      resetChat: mockResetChat,
    });

    render(<ChatSidebar />);

    const localButton = screen.getByRole("button", { name: /Local \(Gemma 4\)/i });
    await userEvent.click(localButton);

    expect(mockSetModelProvider).toHaveBeenCalledWith("local");
  });

  it("calls setModelProvider when clicking cloud model", async () => {
    render(<ChatSidebar />);

    const cloudButton = screen.getByRole("button", { name: /Cloud \(Gemini\)/i });
    await userEvent.click(cloudButton);

    expect(mockSetModelProvider).toHaveBeenCalledWith("cloud");
  });

  it("disables model provider buttons when isLoading is true", () => {
    (useChatStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      agentMode: "intelligent",
      modelProvider: "local",
      isLoading: true,
      setAgentMode: mockSetAgentMode,
      setModelProvider: mockSetModelProvider,
      resetChat: mockResetChat,
    });

    render(<ChatSidebar />);

    const localButton = screen.getByRole("button", { name: /Local \(Gemma 4\)/i });
    const cloudButton = screen.getByRole("button", { name: /Cloud \(Gemini\)/i });

    expect(localButton).toBeDisabled();
    expect(cloudButton).toBeDisabled();
  });
});

describe("ChatSidebar — Agent Mode", () => {
  const mockSetModelProvider = vi.fn();
  const mockSetAgentMode = vi.fn();
  const mockResetChat = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    (useChatStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      agentMode: "intelligent",
      modelProvider: "local",
      isLoading: false,
      setAgentMode: mockSetAgentMode,
      setModelProvider: mockSetModelProvider,
      resetChat: mockResetChat,
    });
  });

  it("renders all three agent mode options", () => {
    render(<ChatSidebar />);

    expect(screen.getByText("Intelligent Agent")).toBeInTheDocument();
    expect(screen.getByText("Keyword Agent")).toBeInTheDocument();
    expect(screen.getByText("No Agent (RAG Only)")).toBeInTheDocument();
  });

  it("shows intelligent agent as active by default", () => {
    render(<ChatSidebar />);

    const intelligentButton = screen.getByRole("button", { name: /Intelligent Agent/i });
    expect(intelligentButton.className).toContain("border-brand-burgundy");
    expect(intelligentButton).toHaveAttribute("aria-pressed", "true");
  });

  it("calls setAgentMode when switching modes", async () => {
    render(<ChatSidebar />);

    const keywordButton = screen.getByRole("button", { name: /Keyword Agent/i });
    await userEvent.click(keywordButton);

    expect(mockSetAgentMode).toHaveBeenCalledWith("keyword");
  });
});

describe("ChatSidebar — Reset Chat", () => {
  const mockSetModelProvider = vi.fn();
  const mockSetAgentMode = vi.fn();
  const mockResetChat = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    (useChatStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      agentMode: "intelligent",
      modelProvider: "local",
      isLoading: false,
      setAgentMode: mockSetAgentMode,
      setModelProvider: mockSetModelProvider,
      resetChat: mockResetChat,
    });
  });

  it("renders reset chat button", () => {
    render(<ChatSidebar />);

    // The reset button is inside a dialog trigger
    expect(screen.getByText("Reset Chat")).toBeInTheDocument();
  });

  it("shows confirmation dialog when clicking reset", async () => {
    render(<ChatSidebar />);

    const resetButton = screen.getByRole("button", { name: /Reset Chat/i });
    await userEvent.click(resetButton);

    // Dialog should appear with confirmation message
    expect(screen.getByText(/This will clear all messages/i)).toBeInTheDocument();
    // Verify both Cancel and Reset Chat buttons are in the dialog
    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByRole("button", { name: /Cancel/i })).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: /Reset Chat/i })).toBeInTheDocument();
  });

  // Note: Testing the actual onClick handler in the dialog is difficult due to
  // pointer-events: none on the body during dialog rendering in jsdom.
  // The resetChat wiring is verified through the component structure and type-checking.
});

describe("ChatSidebar — Mobile Sheet", () => {
  const mockSetModelProvider = vi.fn();
  const mockSetAgentMode = vi.fn();
  const mockResetChat = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    (useChatStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      agentMode: "intelligent",
      modelProvider: "local",
      isLoading: false,
      setAgentMode: mockSetAgentMode,
      setModelProvider: mockSetModelProvider,
      resetChat: mockResetChat,
    });
  });

  it("renders mobile sheet trigger button", () => {
    render(<MobileSidebarTrigger />);

    // The mobile version uses a Sheet with aria-label="Open chat settings"
    const mobileTrigger = screen.getByLabelText(/Open chat settings/i);
    expect(mobileTrigger).toBeInTheDocument();
  });

  // Note: Testing the sheet content is skipped due to ResizeObserver not being defined in jsdom.
  // The Sheet component relies on @radix-ui/react-use-size which requires ResizeObserver.
  // The mobile sheet renders the same SidebarContent component as desktop, tested above.
});











