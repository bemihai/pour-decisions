# Pour Decisions — Frontend

Next.js 16 + TypeScript + Tailwind v4 + shadcn/ui frontend for the Pour Decisions wine RAG application.

## Stack

| Layer        | Technology                                      |
|--------------|-------------------------------------------------|
| Framework    | Next.js 16 (App Router)                         |
| Language     | TypeScript (strict)                             |
| Styling      | Tailwind CSS v4 + shadcn/ui                     |
| Charts       | Recharts                                        |
| Server state | TanStack Query (`@tanstack/react-query`)         |
| Client state | Zustand                                         |
| Testing      | Vitest + React Testing Library                  |
| API          | FastAPI on `:8000` — typed client in `src/lib/api.ts` |

## Setup

```bash
# Install Node.js dependencies
cd frontend
npm install

# Start dev server (requires FastAPI on :8000)
npm run dev
# Or from project root:
make frontend
```

Environment variable (optional — defaults to `http://localhost:8000/api`):
```bash
# frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

## Development

```bash
# From project root — recommended for full stack
make dev-full        # ChromaDB + FastAPI :8000 + Next.js :3000 (hot-reload)

# Frontend only (FastAPI must already be running)
make frontend        # npm run dev on :3000

# Production build
make frontend-build  # npm run build
make run             # build + start full production stack
```

## Pages

| Route           | File                            | Description                          |
|-----------------|---------------------------------|--------------------------------------|
| `/`             | `app/page.tsx`                  | Chat page (wine Q&A, agent modes)    |
| `/cellar`       | `app/cellar/page.tsx`           | Wine cellar inventory + 9 charts     |
| `/taste-profile`| `app/taste-profile/page.tsx`    | Taste profile analytics dashboard    |

## Component Structure

```
src/
  app/
    layout.tsx                  # Root layout: QueryProvider, Navigation, ThemeToggle
    page.tsx                    # Chat page
    cellar/page.tsx             # Cellar page
    taste-profile/page.tsx      # Taste profile page

  components/
    # Shared
    ChatInterface.tsx           # Main chat UI (input + message list)
    ChatMessage.tsx             # Individual message bubble (human/AI/error)
    ChatSidebar.tsx             # Agent mode selector + mobile Sheet drawer
    SourceList.tsx              # RAG / web source citations below AI messages
    MetricCard.tsx              # KPI display card (replaces st.metric())
    DrinkingIndex.tsx           # Drinking readiness indicator with progress bar
    WineCard.tsx                # Expandable wine detail card
    FilterPanel.tsx             # Reusable filter controls (cellar + taste history)
    PageHeader.tsx              # Page title with gradient (compact and full variants)
    Navigation.tsx              # Top navigation bar
    Rating.tsx                  # Star-style rating display
    Section.tsx                 # Collapsible section wrapper
    EmptyState.tsx              # Empty state placeholder

    # Cellar
    cellar/
      CellarOverview.tsx        # 5 KPI metric cards
      CellarTabs.tsx            # Inventory / Statistics tab switcher
      CellarInventory.tsx       # Filterable, sortable wine list
      CellarStatistics.tsx      # 9 Recharts charts
      CellarSyncButton.tsx      # Non-blocking CellarTracker sync

    # Taste Profile
    taste-profile/
      TasteOverview.tsx         # 4 KPI metric cards
      TasteProfileContent.tsx   # 3-tab switcher (Analytics / History / Favorites)
      TasteAnalytics.tsx        # 5 Recharts charts
      TasteHistory.tsx          # Consumed wines with FilterPanel
      TasteFavorites.tsx        # Ranked lists (producers/regions/countries/vintages)

    # Charts
    charts/                     # Recharts wrapper components

    # shadcn/ui primitives
    ui/                         # Button, Card, Badge, Dialog, Select, Tabs, etc.

  lib/
    api.ts                      # Typed fetch() wrappers for every FastAPI endpoint
    types.ts                    # TypeScript interfaces mirroring Pydantic schemas
    utils.ts                    # cn(), getRatingLabel(), getDrinkingStatus(), etc.

  stores/
    chatStore.ts                # Zustand: chat messages, agent mode, loading state
    # (add more stores here for new features)

  test/
    setup.ts                    # @testing-library/jest-dom global setup
```

## Adding a New Page

1. Create `src/app/<route>/page.tsx`.
2. Add navigation link in `src/components/Navigation.tsx`.
3. Add a new FastAPI route in `src/api/routes/<route>.py` and register it in `src/api/main.py`.
4. Add typed API wrappers in `src/lib/api.ts`.
5. Add matching TypeScript interfaces in `src/lib/types.ts`.

## Adding a New Component

1. Create `src/components/<Name>.tsx`.
2. Use `cn()` from `@/lib/utils` for class merging.
3. Use shadcn/ui primitives from `@/components/ui/` where possible.
4. Write a test in `src/components/__tests__/<Name>.test.tsx`.

## Data Flow

- **Server Components** (`page.tsx` files) parallel-fetch initial data via `src/lib/api.ts` at request time.
- **Client Components** (`"use client"`) handle interactivity, filtering, and TanStack Query for live data.
- **Zustand** stores (`src/stores/`) manage client-side session state (chat messages, agent mode, filters).

## Type Sync

TypeScript interfaces in `src/lib/types.ts` mirror Pydantic schemas in `src/api/schemas/`.
Keep these in sync manually when changing request/response shapes.

## Testing

```bash
# Run all tests once and exit (used by CI / make frontend-test)
npm test

# Interactive watch mode for development
npm run test:watch

# Coverage report (outputs to coverage/)
npm run test:coverage
```

Tests live in `src/components/__tests__/`. Each test file mirrors its component:

| Test file                         | Component tested         | What it covers                            |
|-----------------------------------|--------------------------|-------------------------------------------|
| `ChatMessage.test.tsx`            | `ChatMessage`            | Human/AI/error roles, sources, copy button, follow-ups |
| `SourceList.test.tsx`             | `SourceList`             | RAG relevance labels, web source links    |
| `MetricCard.test.tsx`             | `MetricCard`             | Label/value rendering, delta color logic  |
| `DrinkingIndex.test.tsx`          | `DrinkingIndex` + utils  | `getDrinkingStatus()` pure function + component |
| `PageHeader.test.tsx`             | `PageHeader`             | Title, subtitle, compact/full variants    |

Test setup: `src/test/setup.ts` imports `@testing-library/jest-dom`. Vitest config: `vitest.config.ts`.
