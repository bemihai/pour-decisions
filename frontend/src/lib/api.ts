/**
 * Typed API client for the Pour Decisions FastAPI backend.
 *
 * All functions are thin wrappers around fetchJSON<T> and mirror the
 * route handlers in src/api/routes/. The base URL is configured via the
 * NEXT_PUBLIC_API_URL environment variable (default: http://localhost:8000/api).
 */
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
  MergeDecisionResponse,
  MergeEntityType,
  MergeSuggestionsResponse,
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

export function getMergeSuggestions(): Promise<MergeSuggestionsResponse> {
  return fetchJSON<MergeSuggestionsResponse>("/cellar/merge-suggestions");
}

export function mergeSuggestion(
  entityType: MergeEntityType,
  keepId: number,
  removeId: number,
  approve: boolean,
): Promise<MergeDecisionResponse> {
  return fetchJSON<MergeDecisionResponse>(`/cellar/merge/${entityType}/${keepId}/${removeId}`, {
    method: "POST",
    body: JSON.stringify({ approve }),
  });
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
    body: JSON.stringify({}),
  });
}

export function generateProducerDescription(wineId: number): Promise<DescriptionResponse> {
  return fetchJSON<DescriptionResponse>(`/wines/${wineId}/producer-description`, {
    method: "POST",
    body: JSON.stringify({}),
  });
}

