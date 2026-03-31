# API Integration Testing Plan

## Overview

This document covers the manual end-to-end integration test plan for all Pour Decisions API endpoints. Run these after the automated unit tests pass to verify correct behaviour against a live database and (optionally) real LLM.

**Prerequisites:**
```bash
# Terminal 1 – ChromaDB (optional, needed for RAG endpoints)
make chroma-up

# Terminal 2 – FastAPI backend
PYTHONPATH=. uvicorn src.api.main:app --reload --port 8080

# Verify startup
curl http://localhost:8080/health
```

Expected health response:
```json
{
  "status": "ok",
  "resources": {
    "model": true,
    "intelligent_agent": true,
    "keyword_agent": true,
    "retriever": true,
    "reranker": false
  }
}
```

Interactive docs: `http://localhost:8080/docs`

---

## 1. Health & App Shell

| # | Test | Expected |
|---|------|----------|
| 1.1 | `GET /health` | `200`, `status: ok`, all resource keys present |
| 1.2 | `GET /docs` | `200`, Swagger UI renders |
| 1.3 | `GET /openapi.json` | `200`, all four router prefixes present (`/api/chat`, `/api/cellar`, `/api/taste-profile`, `/api/wines`) |
| 1.4 | CORS preflight `OPTIONS /health` with `Origin: http://localhost:3000` | `200`, `access-control-allow-origin` header present |

---

## 2. Cellar Endpoints (`/api/cellar`)

### 2.1 Inventory

```bash
# All wines in cellar (grouped by wine_id)
curl "http://localhost:8080/api/cellar/inventory"

# Filter by type
curl "http://localhost:8080/api/cellar/inventory?wine_type=Red"

# Filter by country + search
curl "http://localhost:8080/api/cellar/inventory?country=France&search=pinot"

# Rating filter
curl "http://localhost:8080/api/cellar/inventory?rating_filter=90%2B"

# Sort by drinking window readiness (descending)
curl "http://localhost:8080/api/cellar/inventory?sort_by=drink_desc"
```

**Acceptance criteria:**
- Response matches `InventoryResponse` schema: `items`, `total_wines`, `total_bottles`, `filter_options`
- `total_bottles` equals sum of `quantity` across all `items`
- `filter_options.wine_types` contains every distinct type in the full (unfiltered) inventory
- Filtered results respect the applied filter
- Sort order is correct

### 2.2 Filters

```bash
curl "http://localhost:8080/api/cellar/filters"
```

**Acceptance criteria:** `FilterOptions` schema with populated `wine_types`, `countries`, `locations`, `producers`, sensible `min_vintage`/`max_vintage`.

### 2.3 Stats

```bash
curl "http://localhost:8080/api/cellar/stats"
```

**Acceptance criteria:**
- `overview.total_bottles` matches actual bottle count
- `drinking_stats` fields (`ready_to_drink`, `to_hold`, `unknown`) sum to approximately `total_bottles`
- `value_stats.by_currency` lists currencies with price data

### 2.4 Charts

```bash
curl "http://localhost:8080/api/cellar/charts"
```

**Acceptance criteria:** `ChartDataResponse` with non-empty `wine_type_distribution`, `country_distribution`, `varietal_distribution`.

### 2.5 CellarTracker Sync

```bash
curl -X POST "http://localhost:8080/api/cellar/sync"
```

**Acceptance criteria:**
- With valid credentials in `.env`: `200`, `success: true`, non-zero counts
- Without credentials: `400`, detail mentions "credentials"
- Network error: `200`, `success: false`, `error_message` set

---

## 3. Taste Profile Endpoints (`/api/taste-profile`)

### 3.1 Overview

```bash
curl "http://localhost:8080/api/taste-profile/overview"
```

**Acceptance criteria:** `avg_rating` is in range 0-100, `wines_rated` > 0 if tastings exist, `favorite_type` is a valid wine type string.

### 3.2 Rating Distribution

```bash
curl "http://localhost:8080/api/taste-profile/rating-distribution"
```

**Acceptance criteria:**
- `buckets` is non-empty if consumed wines with ratings exist
- `total` equals sum of all `bucket.count` values
- All `bucket.color` values are valid CSS `rgb(...)` or hex strings
- Buckets are in ascending rating order

### 3.3 Wine Types

```bash
curl "http://localhost:8080/api/taste-profile/wine-types"
```

**Acceptance criteria:** `types` list with `wine_type`, `wines_tasted`, optional `avg_rating`.

### 3.4 Varietals

```bash
# Default limit (10)
curl "http://localhost:8080/api/taste-profile/varietals"

# Custom limit
curl "http://localhost:8080/api/taste-profile/varietals?limit=5"

# Invalid limit
curl "http://localhost:8080/api/taste-profile/varietals?limit=0"   # expect 422
curl "http://localhost:8080/api/taste-profile/varietals?limit=51"  # expect 422
```

### 3.5 Producers

```bash
curl "http://localhost:8080/api/taste-profile/producers"
curl "http://localhost:8080/api/taste-profile/producers?limit=10"
```

### 3.6 Regions

```bash
curl "http://localhost:8080/api/taste-profile/regions"
```

### 3.7 Countries

```bash
curl "http://localhost:8080/api/taste-profile/countries"
```

### 3.8 Vintages

```bash
curl "http://localhost:8080/api/taste-profile/vintages"
```

**Note:** Only vintages with ≥ 2 bottles consumed are returned.

### 3.9 Appellations

```bash
curl "http://localhost:8080/api/taste-profile/appellations"
```

### 3.10 Rating Trends

```bash
curl "http://localhost:8080/api/taste-profile/rating-trends"
```

**Acceptance criteria:**
- `trend` is one of `improving`, `declining`, `stable`
- `points` has at most 12 entries (last 12 months)
- `points[].month` is in `YYYY-MM` format

### 3.11 Consumed Wines

```bash
# All consumed
curl "http://localhost:8080/api/taste-profile/consumed"

# With filters
curl "http://localhost:8080/api/taste-profile/consumed?wine_type=Red&rating_filter=90%2B"
curl "http://localhost:8080/api/taste-profile/consumed?search=barolo&sort_by=rating_desc"
curl "http://localhost:8080/api/taste-profile/consumed?rating_filter=unrated"
curl "http://localhost:8080/api/taste-profile/consumed?min_vintage=2015&max_vintage=2020"

# Limit
curl "http://localhost:8080/api/taste-profile/consumed?limit=5"
```

**Acceptance criteria:**
- `total` reflects the count before `limit` is applied
- `items` length ≤ `limit`
- `filter_options` reflects the full unfiltered set, not the filtered result
- `rating_description` is non-null for rated wines

---

## 4. Wine Detail Endpoints (`/api/wines`)

### 4.1 Get Wine Detail

```bash
# Known wine (replace 1 with a real ID from /api/cellar/inventory)
curl "http://localhost:8080/api/wines/1"

# Non-existent wine
curl "http://localhost:8080/api/wines/999999"   # expect 404
```

**Acceptance criteria:**
- `bottles` list with correct `status`, `location`, `bin` values
- `owned_quantity` equals sum of `quantity` where `status == "in_cellar"`
- `producer_description` present when producer has a cached description
- 404 detail contains the wine ID

### 4.2 Generate Description

```bash
# Default (RAG on, web search off)
curl -X POST "http://localhost:8080/api/wines/1/description" \
  -H "Content-Type: application/json" \
  -d '{}'

# RAG off
curl -X POST "http://localhost:8080/api/wines/1/description" \
  -H "Content-Type: application/json" \
  -d '{"use_rag_context": false}'

# Non-existent wine
curl -X POST "http://localhost:8080/api/wines/999999/description"  # expect 404

# Call again immediately (should re-generate since description is cleared)
curl -X POST "http://localhost:8080/api/wines/1/description"
```

**Acceptance criteria:**
- `success: true`, `description` is a non-empty string (2-3 sentences)
- `drink_from_year` / `drink_to_year` populated when LLM estimates drinking window
- Subsequent `GET /api/wines/1` returns the new description
- On LLM failure: `success: false`, `error` field set, status still `200`

---

## 5. Chat Endpoints (`/api/chat`)

### 5.1 Initial Message

```bash
curl "http://localhost:8080/api/chat/initial-message"
```

**Acceptance criteria:** `role: "ai"`, non-empty `content`.

### 5.2 Chat – Intelligent Agent

```bash
curl -X POST "http://localhost:8080/api/chat/" \
  -H "Content-Type: application/json" \
  -d '{"message": "What wines are ready to drink in my cellar?", "agent_mode": "intelligent"}'
```

**Acceptance criteria:**
- `answer` references cellar contents (agent used cellar tool)
- `agent_mode: "intelligent"`
- `error` is null

### 5.3 Chat – Keyword Agent

```bash
curl -X POST "http://localhost:8080/api/chat/" \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about Barolo wine", "agent_mode": "keyword"}'
```

**Acceptance criteria:** `answer` non-empty, `agent_mode: "keyword"`.

### 5.4 Chat – RAG Only

```bash
curl -X POST "http://localhost:8080/api/chat/" \
  -H "Content-Type: application/json" \
  -d '{"message": "What grapes are used in Burgundy?", "agent_mode": "rag_only"}'
```

**Acceptance criteria:** `answer` mentions Pinot Noir, `sources` non-empty when ChromaDB has wine book data.

### 5.5 Chat – With History

```bash
curl -X POST "http://localhost:8080/api/chat/" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "And what about pairing it with food?",
    "agent_mode": "rag_only",
    "message_history": [
      {"role": "human", "content": "Tell me about Barolo"},
      {"role": "ai", "content": "Barolo is a full-bodied red from Piedmont..."}
    ]
  }'
```

**Acceptance criteria:** Answer is contextually aware of Barolo (not generic).

### 5.6 Chat – Error Cases

```bash
# Empty message
curl -X POST "http://localhost:8080/api/chat/" \
  -H "Content-Type: application/json" \
  -d '{"message": "", "agent_mode": "rag_only"}'   # expect 422

# Agent unavailable (intelligent with no agent loaded)
# Temporarily set GOOGLE_API_KEY= to disable LLM loading, then:
curl -X POST "http://localhost:8080/api/chat/" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "agent_mode": "intelligent"}'  # expect 503
```

---

## 6. Automated Unit Test Coverage Summary

| File | Tests | Coverage Focus |
|------|-------|----------------|
| `tests/api/test_app.py` | 5 | Health endpoint, OpenAPI schema, CORS, router registration |
| `tests/api/test_cellar.py` | 19 | Inventory grouping/filtering/sorting, stats, charts, sync credentials, sync failure |
| `tests/api/test_taste_profile.py` | 29 | All 11 endpoints, empty states, limit validation, filter options, trend directions, sort orders |
| `tests/api/test_wines.py` | 10 | Wine detail assembly, 404, bottle list, producer description, description generation flags, error handling |
| `tests/api/test_chat.py` | 13 | All three agent modes, 503 on missing agent, 422 on empty message, history forwarding, quota error message |

**Total: 76 unit tests, all passing.**

Run:
```bash
PYTHONPATH=. python -m pytest tests/api/ -v
```

---

## 7. Acceptance Criteria Summary

Every endpoint must:
1. Return HTTP `200` for valid requests (exception: `404` for missing resources, `422` for invalid params, `503` for unavailable agents).
2. Return valid JSON matching its declared Pydantic schema.
3. Return empty-but-valid responses (not errors) when the database has no data.
4. Not expose raw exception tracebacks in error responses.
5. Return filter options from the **full unfiltered set** on paginated/filtered endpoints.

