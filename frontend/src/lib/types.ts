/**
 * TypeScript interfaces mirroring the FastAPI Pydantic schemas.
 * Keep in sync with: src/api/schemas/{chat,cellar,taste_profile,wines}.py
 */

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

export type AgentMode = "intelligent" | "keyword" | "rag_only";

export type ModelProvider = "local" | "cloud";

export interface ChatMessage {
  role: "human" | "ai";
  content: string;
}

export interface ChatRequest {
  message: string;
  agent_mode?: AgentMode;
  model_provider?: ModelProvider;
  message_history?: ChatMessage[];
  enable_rag?: boolean;
  n_results?: number | null;
}

export interface Source {
  name: string;
  page: number | null;
  relevance: number | null;
}

export interface WebSource {
  title: string;
  url: string;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
  web_sources: WebSource[];
  agent_mode: AgentMode;
  model_provider?: ModelProvider;
  error: string | null;
}

export interface InitialMessageResponse {
  role: string;
  content: string;
}

// ---------------------------------------------------------------------------
// Cellar — inventory
// ---------------------------------------------------------------------------

export interface FilterOptions {
  wine_types: string[];
  countries: string[];
  locations: string[];
  producers: string[];
  min_vintage: number;
  max_vintage: number;
}

export interface InventoryItem {
  wine_id: number;
  wine_name: string;
  producer_name: string | null;
  vintage: number | null;
  wine_type: string | null;
  varietal: string | null;
  country: string | null;
  region_name: string | null;
  quantity: number;
  personal_rating: number | null;
  community_rating: number | null;
  drink_index: number | null;
  drink_from_year: number | null;
  drink_to_year: number | null;
  drink_window_source: string | null;
  location: string | null;
  bin: string | null;
  purchase_price: number | null;
  currency: string | null;
  description: string | null;
  producer_description: string | null;
  producer_id: number | null;
  bottle_note: string | null;
  last_tasted_date: string | null;
  like_votes: number | null;
  like_percentage: number | null;
  q_purchased: number;
  q_consumed: number;
  q_quantity: number;
  created_at: string | null;
}

export interface InventoryResponse {
  items: InventoryItem[];
  total_wines: number;
  total_bottles: number;
  filter_options: FilterOptions;
}

/** Query parameters accepted by GET /api/cellar/inventory */
export interface InventoryFilters {
  wine_type?: string;
  country?: string;
  producer?: string;
  location?: string;
  min_vintage?: number;
  max_vintage?: number;
  rating_filter?: string;
  search?: string;
  sort_by?: string;
}

// ---------------------------------------------------------------------------
// Cellar — stats
// ---------------------------------------------------------------------------

export interface WineTypeCount {
  wine_type: string;
  unique_wines: number;
  bottles: number;
}

export interface CountryCount {
  country: string | null;
  unique_wines: number;
  bottles: number;
}

export interface CellarOverview {
  total_bottles: number;
  unique_wines: number;
  by_type: WineTypeCount[];
  by_country: CountryCount[];
}

export interface DrinkingWindowStats {
  ready_to_drink: number;
  to_hold: number;
  unknown: number;
}

export interface CurrencyValue {
  currency: string;
  total_value: number;
  wines_with_price: number;
}

export interface CellarValueStats {
  by_currency: CurrencyValue[];
  bottles_without_price: number;
}

export interface CellarStatsResponse {
  overview: CellarOverview;
  drinking_stats: DrinkingWindowStats;
  value_stats: CellarValueStats;
}

// ---------------------------------------------------------------------------
// Cellar — charts
// ---------------------------------------------------------------------------

export interface ChartDataResponse {
  wine_type_distribution: Record<string, unknown>[];
  country_distribution: Record<string, unknown>[];
  varietal_distribution: Record<string, unknown>[];
  region_distribution: Record<string, unknown>[];
  drinking_window_wines: Record<string, unknown>;
  cellar_size_over_time: Record<string, unknown>[];
  top_rated: Record<string, unknown>[];
  /** Bottle counts by vintage year for all in-cellar wines. */
  vintage_distribution: Record<string, unknown>[];
  /** Personal-rating tier counts for rated in-cellar wines. */
  rating_distribution: Record<string, unknown>[];
  /** Bottle counts bucketed by wine age range. */
  wine_age_distribution: Record<string, unknown>[];
}

// ---------------------------------------------------------------------------
// Cellar — sync
// ---------------------------------------------------------------------------

export interface SyncResponse {
  success: boolean;
  wines_processed: number;
  wines_imported: number;
  bottles_processed: number;
  bottles_imported: number;
  producers_created: number;
  regions_created: number;
  errors: string[];
  error_message: string | null;
}

// ---------------------------------------------------------------------------
// Taste profile — overview
// ---------------------------------------------------------------------------

export interface TasteOverviewResponse {
  avg_rating: number | null;
  wines_rated: number;
  favorite_type: string;
  highly_rated_count: number;
  highly_rated_pct: number;
}

// ---------------------------------------------------------------------------
// Taste profile — rating distribution
// ---------------------------------------------------------------------------

export interface RatingBucket {
  range: string;
  count: number;
}

export interface RatingDistributionResponse {
  buckets: RatingBucket[];
  total: number;
}

// ---------------------------------------------------------------------------
// Taste profile — wine types
// ---------------------------------------------------------------------------

export interface WineTypeStats {
  wine_type: string;
  wines_tasted: number;
  avg_rating: number | null;
  highest_rating: number | null;
  most_recent_date: string | null;
}

export interface WineTypesResponse {
  types: WineTypeStats[];
}

// ---------------------------------------------------------------------------
// Taste profile — varietals
// ---------------------------------------------------------------------------

export interface VarietalStats {
  varietal: string;
  wines_tasted: number;
  avg_rating: number | null;
  highest_rating: number | null;
}

export interface VarietalsResponse {
  varietals: VarietalStats[];
}

// ---------------------------------------------------------------------------
// Taste profile — producers
// ---------------------------------------------------------------------------

export interface ProducerStats {
  producer_name: string;
  country: string | null;
  wines_tasted: number;
  avg_rating: number | null;
  highest_rating: number | null;
}

export interface ProducersResponse {
  producers: ProducerStats[];
}

// ---------------------------------------------------------------------------
// Taste profile — regions
// ---------------------------------------------------------------------------

export interface RegionStats {
  region_name: string;
  country: string | null;
  wines_tasted: number;
  avg_rating: number | null;
  highest_rating: number | null;
}

export interface RegionsResponse {
  regions: RegionStats[];
}

// ---------------------------------------------------------------------------
// Taste profile — countries
// ---------------------------------------------------------------------------

export interface CountryStats {
  country: string;
  wines_tasted: number;
  avg_rating: number | null;
  highest_rating: number | null;
}

export interface CountriesResponse {
  countries: CountryStats[];
}

// ---------------------------------------------------------------------------
// Taste profile — vintages
// ---------------------------------------------------------------------------

export interface VintageStats {
  vintage: number;
  wines_tasted: number;
  avg_rating: number | null;
  highest_rating: number | null;
}

export interface VintagesResponse {
  vintages: VintageStats[];
}

// ---------------------------------------------------------------------------
// Taste profile — appellations
// ---------------------------------------------------------------------------

export interface AppellationStats {
  appellation: string;
  country: string | null;
  wines_tasted: number;
  avg_rating: number | null;
  highest_rating: number | null;
}

export interface AppellationsResponse {
  appellations: AppellationStats[];
}

// ---------------------------------------------------------------------------
// Taste profile — rating trends
// ---------------------------------------------------------------------------

export interface RatingTrendPoint {
  month: string;
  avg_rating: number;
  wines_count: number;
}

export interface RatingTrendsResponse {
  points: RatingTrendPoint[];
  trend: string | null;
}

// ---------------------------------------------------------------------------
// Taste profile — consumed wines
// ---------------------------------------------------------------------------

export interface ConsumedWineItem {
  wine_id: number | null;
  wine_name: string;
  producer_name: string | null;
  vintage: number | null;
  wine_type: string | null;
  varietal: string | null;
  country: string | null;
  region_name: string | null;
  consumed_date: string | null;
  personal_rating: number | null;
  community_rating: number | null;
  rating_description: string | null;
  tasting_notes: string | null;
}

export interface ConsumedFilterOptions {
  wine_types: string[];
  countries: string[];
  producers: string[];
  min_vintage: number;
  max_vintage: number;
}

export interface ConsumedWinesResponse {
  items: ConsumedWineItem[];
  total: number;
  filter_options: ConsumedFilterOptions;
}

/** Query parameters accepted by GET /api/taste-profile/consumed */
export interface ConsumedWinesFilters {
  wine_type?: string;
  country?: string;
  producer?: string;
  min_vintage?: number;
  max_vintage?: number;
  rating_filter?: string;
  search?: string;
  sort_by?: string;
  limit?: number;
}

// ---------------------------------------------------------------------------
// Wine detail
// ---------------------------------------------------------------------------

export interface BottleDetail {
  id: number;
  quantity: number;
  status: string;
  location: string | null;
  bin: string | null;
  purchase_date: string | null;
  purchase_price: number | null;
  valuation_price: number | null;
  currency: string;
  store_name: string | null;
  consumed_date: string | null;
  bottle_note: string | null;
}

export interface WineDetailResponse {
  id: number;
  source: string;
  external_id: string | null;
  wine_name: string;
  vintage: number | null;
  wine_type: string;
  varietal: string | null;
  designation: string | null;
  appellation: string | null;
  vineyard: string | null;
  bottle_size: string;
  drink_from_year: number | null;
  drink_to_year: number | null;
  drink_index: number | null;
  drink_window_source: string | null;
  description: string | null;
  producer_description: string | null;
  producer_id: number | null;
  producer_name: string | null;
  region_id: number | null;
  region_name: string | null;
  country: string | null;
  personal_rating: number | null;
  community_rating: number | null;
  tasting_notes: string | null;
  last_tasted_date: string | null;
  q_purchased: number;
  q_quantity: number;
  q_consumed: number;
  bottles: BottleDetail[];
  owned_quantity: number;
  created_at: string | null;
  updated_at: string | null;
}

/** Response from POST /api/wines/:id/description */
export interface DescriptionResponse {
  success: boolean;
  description: string | null;
  drink_from_year: number | null;
  drink_to_year: number | null;
}

