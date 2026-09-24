/**
 * Typed API client for the Pour Decisions FastAPI backend.
 *
 * All functions are thin wrappers around fetchJSON<T> and mirror the
 * route handlers in src/api/routes/. The base URL is configured via the
 * NEXT_PUBLIC_API_URL environment variable (default: http://localhost:8000/api).
 */
import { createParser, type EventSourceMessage } from "eventsource-parser";

import type {
  ChatRequest,
  ChatResponse,
  InitialMessageResponse,
  InventoryFilters,
  InventoryResponse,
  FilterOptions,
  CellarStatsResponse,
  ChartDataResponse,
  DrinkNextResponse,
  SyncResponse,
  TasteOverviewResponse,
  RatingDistributionResponse,
  WineTypesResponse,
  VarietalsResponse,
  ProducersResponse,
  RegionsResponse,
  CountriesResponse,
  VintagesResponse,
  AppellationsResponse,
  RatingTrendsResponse,
  ConsumedWinesFilters,
  ConsumedWinesResponse,
  WineDetailResponse,
  DescriptionResponse,
  AgentDoneStreamEvent,
  ChatStreamEvent,
  StreamErrorEvent,
  ToolProgressStreamEvent,
} from "./types";

// On the server (SSR), use the internal Docker service URL (API_URL) so
// requests reach the api container via the Docker network instead of loopback.
// In the browser, use the public-facing NEXT_PUBLIC_API_URL.
const DEFAULT_API_URL = "http://localhost:8000/api";
const API_BASE =
  typeof window === "undefined"
    ? (process.env.API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? DEFAULT_API_URL)
    : (process.env.NEXT_PUBLIC_API_URL ?? DEFAULT_API_URL);

// ---------------------------------------------------------------------------
// Core fetch wrapper
// ---------------------------------------------------------------------------

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export class ChatStreamError extends Error {
  readonly outcome = "uncertain" as const;

  constructor(message: string) {
    super(message);
    this.name = "ChatStreamError";
  }
}

async function fetchJSON<T>(path: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      message = (body as { detail?: string; message?: string }).detail ?? body.message ?? message;
    } catch {
      // keep the default message
    }
    throw new ApiError(res.status, message);
  }

  return res.json() as Promise<T>;
}

/** Build a query string from a plain object, omitting undefined/null/"" values. */
function toQueryString(params: Record<string, unknown>): string {
  const qs = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      qs.set(key, String(value));
    }
  }
  const str = qs.toString();
  return str ? `?${str}` : "";
}

// ---------------------------------------------------------------------------
// Chat  — POST /api/chat, GET /api/chat/initial-message
// ---------------------------------------------------------------------------

export function sendChatMessage(req: ChatRequest): Promise<ChatResponse> {
  return fetchJSON<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

const STREAM_ERROR_MESSAGE =
  "The request outcome is uncertain because the streaming response could not be completed." as const;
const STREAM_CONTENT_TYPE = "text/event-stream";
const MAX_STREAM_BUFFER_SIZE = 1024 * 1024;

export interface ChatStreamOptions {
  signal?: AbortSignal;
  onProgress?: (event: ToolProgressStreamEvent) => void;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasOnlyKeys(value: Record<string, unknown>, keys: readonly string[]): boolean {
  const allowed = new Set(keys);
  return Object.keys(value).every((key) => allowed.has(key));
}

function isNullableString(value: unknown): value is string | null | undefined {
  return value === undefined || value === null || typeof value === "string";
}

function isChatResponse(value: unknown): value is ChatResponse {
  if (!isRecord(value)) return false;
  return (
    typeof value.answer === "string" &&
    Array.isArray(value.sources) &&
    value.sources.every(
      (source) =>
        isRecord(source) &&
        typeof source.name === "string" &&
        (source.page === null || typeof source.page === "number") &&
        (source.relevance === null || typeof source.relevance === "number"),
    ) &&
    Array.isArray(value.web_sources) &&
    value.web_sources.every(
      (source) =>
        isRecord(source) && typeof source.title === "string" && typeof source.url === "string",
    ) &&
    (value.agent_mode === "intelligent" || value.agent_mode === "rag_only") &&
    (value.model_provider === undefined ||
      value.model_provider === "local" ||
      value.model_provider === "cloud") &&
    isNullableString(value.error) &&
    isNullableString(value.trace_id) &&
    isNullableString(value.thread_id)
  );
}

function parseStreamEvent(message: EventSourceMessage): ChatStreamEvent {
  let value: unknown;
  try {
    value = JSON.parse(message.data);
  } catch {
    throw new ChatStreamError("The streaming response contained malformed JSON.");
  }
  if (!isRecord(value) || message.event !== value.type) {
    throw new ChatStreamError("The streaming response contained a mismatched event.");
  }

  if (
    value.type === "tool_progress" &&
    hasOnlyKeys(value, ["type", "invocation_id", "tool_key", "status"]) &&
    Number.isInteger(value.invocation_id) &&
    (value.invocation_id as number) > 0 &&
    typeof value.tool_key === "string" &&
    value.tool_key.length > 0 &&
    value.tool_key.length <= 128 &&
    (value.status === "started" || value.status === "completed" || value.status === "failed")
  ) {
    return value as unknown as ToolProgressStreamEvent;
  }

  if (
    value.type === "agent_done" &&
    hasOnlyKeys(value, ["type", "response"]) &&
    isChatResponse(value.response)
  ) {
    return value as unknown as AgentDoneStreamEvent;
  }

  if (
    value.type === "stream_error" &&
    hasOnlyKeys(value, ["type", "message", "outcome"]) &&
    value.message === STREAM_ERROR_MESSAGE &&
    value.outcome === "uncertain"
  ) {
    return value as unknown as StreamErrorEvent;
  }

  throw new ChatStreamError("The streaming response did not match the expected contract.");
}

async function readErrorResponse(res: Response): Promise<{ code?: string; message: string }> {
  const fallbackMessage = `HTTP ${res.status}`;
  try {
    const body = (await res.json()) as {
      detail?: string | { code?: string; message?: string };
      message?: string;
    };
    if (isRecord(body.detail)) {
      return {
        code: typeof body.detail.code === "string" ? body.detail.code : undefined,
        message:
          typeof body.detail.message === "string" ? body.detail.message : fallbackMessage,
      };
    }
    return {
      message:
        typeof body.detail === "string"
          ? body.detail
          : typeof body.message === "string"
            ? body.message
            : fallbackMessage,
    };
  } catch {
    return { message: fallbackMessage };
  }
}

/** Execute one chat turn over POST SSE, with safe pre-execution fallback only. */
export async function streamChatMessage(
  req: ChatRequest,
  options: ChatStreamOptions = {},
): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal: options.signal,
  });

  if (!res.ok) {
    const error = await readErrorResponse(res);
    const mayFallback =
      (res.status === 404 && error.code === "streaming_disabled") ||
      (res.status === 400 && error.code === "stream_mode_unsupported");
    if (mayFallback) return sendChatMessage(req);
    throw new ApiError(res.status, error.message);
  }

  const contentType = res.headers.get("content-type")?.toLowerCase() ?? "";
  if (!contentType.startsWith(STREAM_CONTENT_TYPE) || !res.body) {
    throw new ChatStreamError("The server did not return a readable event stream.");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let terminal: AgentDoneStreamEvent | StreamErrorEvent | null = null;
  let readerFinished = false;
  const parser = createParser({
    maxBufferSize: MAX_STREAM_BUFFER_SIZE,
    onError: () => {
      throw new ChatStreamError("The streaming response was malformed.");
    },
    onEvent: (message) => {
      if (terminal !== null) {
        throw new ChatStreamError("The streaming response continued after its terminal event.");
      }
      const event = parseStreamEvent(message);
      if (event.type === "tool_progress") {
        options.onProgress?.(event);
      } else {
        terminal = event;
      }
    },
  });

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        readerFinished = true;
        parser.feed(decoder.decode());
        parser.reset({ consume: true });
        break;
      }
      parser.feed(decoder.decode(value, { stream: true }));
    }

    const completed = terminal as AgentDoneStreamEvent | StreamErrorEvent | null;
    if (completed?.type === "agent_done") return completed.response;
    if (completed?.type === "stream_error") throw new ChatStreamError(completed.message);
    throw new ChatStreamError("The streaming response ended before a terminal event arrived.");
  } finally {
    if (!readerFinished) await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}

export async function deleteChatThread(threadId: string): Promise<void> {
  const url = `${API_BASE}/chat/threads/${encodeURIComponent(threadId)}`;
  const res = await fetch(url, { method: "DELETE" });

  if (!res.ok) {
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      message = (body as { detail?: string; message?: string }).detail ?? body.message ?? message;
    } catch {
      // keep the default message
    }
    throw new ApiError(res.status, message);
  }
}

export function getInitialMessage(): Promise<InitialMessageResponse> {
  return fetchJSON<InitialMessageResponse>("/chat/initial-message");
}

// ---------------------------------------------------------------------------
// Cellar inventory  — GET /api/cellar/inventory, /filters
// ---------------------------------------------------------------------------

export function getInventory(filters?: InventoryFilters): Promise<InventoryResponse> {
  const qs = filters ? toQueryString(filters as Record<string, unknown>) : "";
  return fetchJSON<InventoryResponse>(`/cellar/inventory${qs}`);
}

export function getFilterOptions(): Promise<FilterOptions> {
  return fetchJSON<FilterOptions>("/cellar/filters");
}

// ---------------------------------------------------------------------------
// Cellar stats & charts  — GET /api/cellar/stats, /charts
// ---------------------------------------------------------------------------

export function getCellarStats(): Promise<CellarStatsResponse> {
  return fetchJSON<CellarStatsResponse>("/cellar/stats");
}

export function getCellarCharts(): Promise<ChartDataResponse> {
  return fetchJSON<ChartDataResponse>("/cellar/charts");
}

export function getDrinkNext(limit = 50): Promise<DrinkNextResponse> {
  return fetchJSON<DrinkNextResponse>(`/cellar/drink-next?limit=${limit}`);
}

// ---------------------------------------------------------------------------
// Cellar sync  — POST /api/cellar/sync
// ---------------------------------------------------------------------------

export function syncCellarTracker(): Promise<SyncResponse> {
  return fetchJSON<SyncResponse>("/cellar/sync", { method: "POST" });
}


// ---------------------------------------------------------------------------
// Taste profile overview  — GET /api/taste-profile/overview
// ---------------------------------------------------------------------------

export function getTasteOverview(): Promise<TasteOverviewResponse> {
  return fetchJSON<TasteOverviewResponse>("/taste-profile/overview");
}

// ---------------------------------------------------------------------------
// Taste profile charts
// ---------------------------------------------------------------------------

export function getRatingDistribution(): Promise<RatingDistributionResponse> {
  return fetchJSON<RatingDistributionResponse>("/taste-profile/rating-distribution");
}

export function getWineTypes(): Promise<WineTypesResponse> {
  return fetchJSON<WineTypesResponse>("/taste-profile/wine-types");
}

export function getVarietals(limit = 10): Promise<VarietalsResponse> {
  return fetchJSON<VarietalsResponse>(`/taste-profile/varietals?limit=${limit}`);
}

export function getProducers(limit = 5): Promise<ProducersResponse> {
  return fetchJSON<ProducersResponse>(`/taste-profile/producers?limit=${limit}`);
}

export function getRegions(limit = 5): Promise<RegionsResponse> {
  return fetchJSON<RegionsResponse>(`/taste-profile/regions?limit=${limit}`);
}

export function getCountries(limit = 5): Promise<CountriesResponse> {
  return fetchJSON<CountriesResponse>(`/taste-profile/countries?limit=${limit}`);
}

export function getVintages(limit = 5): Promise<VintagesResponse> {
  return fetchJSON<VintagesResponse>(`/taste-profile/vintages?limit=${limit}`);
}

export function getAppellations(limit = 5): Promise<AppellationsResponse> {
  return fetchJSON<AppellationsResponse>(`/taste-profile/appellations?limit=${limit}`);
}

export function getRatingTrends(): Promise<RatingTrendsResponse> {
  return fetchJSON<RatingTrendsResponse>("/taste-profile/rating-trends");
}

// ---------------------------------------------------------------------------
// Taste profile consumed wines  — GET /api/taste-profile/consumed
// ---------------------------------------------------------------------------

export function getConsumedWines(filters?: ConsumedWinesFilters): Promise<ConsumedWinesResponse> {
  const qs = filters ? toQueryString(filters as Record<string, unknown>) : "";
  return fetchJSON<ConsumedWinesResponse>(`/taste-profile/consumed${qs}`);
}

// ---------------------------------------------------------------------------
// Wine detail  — GET /api/wines/:id, POST /api/wines/:id/description
// ---------------------------------------------------------------------------

export function getWine(wineId: number): Promise<WineDetailResponse> {
  return fetchJSON<WineDetailResponse>(`/wines/${wineId}`);
}

export function generateWineDescription(wineId: number): Promise<DescriptionResponse> {
  return fetchJSON<DescriptionResponse>(`/wines/${wineId}/description`, {
    method: "POST",
    body: JSON.stringify({ use_rag_context: true, use_web_search: true }),
  });
}

export function generateProducerDescription(wineId: number): Promise<DescriptionResponse> {
  return fetchJSON<DescriptionResponse>(`/wines/${wineId}/producer-description`, {
    method: "POST",
    body: JSON.stringify({ use_rag_context: true, use_web_search: true }),
  });
}
