# Pour Decisions — Frontend

Next.js 16 + TypeScript + Tailwind v4 + shadcn/ui frontend for the Pour Decisions wine RAG application.

## Stack

| Layer | Technology |
|---|---|
| Framework | Next.js 16 (App Router, Turbopack) |
| Language | TypeScript (strict) |
| Styling | Tailwind CSS v4 + shadcn/ui |
| Charts | Plotly.js (via `PlotlyChart` wrapper) |
| Server state | TanStack Query (`@tanstack/react-query`) |
| Client state | Zustand + persist |
| API | FastAPI on `:8000` — typed client in `src/lib/api.ts` |

## Pages

| Route | File | Description |
|---|---|---|
| `/` | `app/page.tsx` | Chat page (Phase 1) |
| `/cellar` | `app/cellar/page.tsx` | Wine cellar inventory + 9 charts (Phase 2) |
| `/taste-profile` | `app/taste-profile/page.tsx` | Taste profile analytics (Phase 3) |

## Component tree

```
src/
  components/
    # Shared
    MetricCard.tsx          — KPI display card (replaces st.metric())
    PlotlyChart.tsx         — Plotly.js wrapper (all charts)
    FilterPanel.tsx         — Reusable filter controls (cellar + taste history)
    WineCard.tsx            — Expandable wine detail card
    DrinkingIndex.tsx       — Drinking readiness indicator
    PageHeader.tsx          — Page title with gradient
    Navigation.tsx          — Top nav bar with dark mode toggle
    ThemeToggle.tsx         — Light/dark mode switcher

    # Chat (Phase 1)
    ChatInterface.tsx       — Main chat UI
    ChatMessage.tsx         — Individual message bubble
    ChatSidebar.tsx         — Agent mode selector + mobile Sheet drawer
    SourceList.tsx          — RAG / web source citations

    # Cellar (Phase 2)
    cellar/
      CellarOverview.tsx    — 5 KPI metrics
      CellarTabs.tsx        — Inventory / Statistics tab switcher
      CellarInventory.tsx   — Filterable wine list
      CellarStatistics.tsx  — 9 Plotly charts
      CellarSyncButton.tsx  — Non-blocking CellarTracker sync

    # Taste Profile (Phase 3)
    taste-profile/
      TasteOverview.tsx     — 4 KPI metrics
      TasteProfileContent.tsx — 3-tab switcher (Analytics/History/Favorites)
      TasteAnalytics.tsx    — 5 Plotly charts
      TasteHistory.tsx      — Consumed wines with FilterPanel + TanStack Query
      TasteFavorites.tsx    — Ranked lists (producers/regions/countries/vintages/appellations)
```

## Development

```bash
# From project root
make frontend        # npm run dev on :3000
make dev-full        # ChromaDB + FastAPI + Next.js together
```

Environment variable: `NEXT_PUBLIC_API_URL` (default `http://localhost:8000/api`).

## Data flow

- **Server Components** (`page.tsx` files) parallel-fetch data via `src/lib/api.ts` at request time.
- **Client Components** (`"use client"`) handle interactivity, filtering, and TanStack Query caching.
- **Zustand** stores (`src/stores/`) manage client-side session state (chat messages, agent mode).

## Type sync

TypeScript interfaces in `src/lib/types.ts` mirror Pydantic schemas in `src/api/schemas/`.
Keep these in sync manually when changing request/response shapes.
