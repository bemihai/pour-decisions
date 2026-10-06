# Frontend Agent Instructions

> **Project version**: 0.12.0 — last updated 2026-10-06.

Follow the authoritative repository policy in `../AGENTS.md`, including approval continuity.

## UI Direction

- The user is primarily a backend engineer. For meaningful frontend changes, explain the intended
  UI direction and request approval before implementation unless it is already approved.
- Keep the UI functional at all times.
- Do not perform major frontend refactors without a reviewed plan and explicit approval.
- Creative UI work is allowed only when the rationale is clear and usability remains intact.

## Implementation Conventions

- Use Next.js App Router, strict TypeScript, Tailwind CSS v4, and the existing shadcn/ui components.
- The typed API client is `src/lib/api.ts`; keep `src/lib/types.ts` aligned with backend API schemas.
- TanStack Query manages server state; Zustand stores in `src/stores/` manage client state.
- Streaming uses fetch-based POST SSE. Fall back only for explicit pre-execution disabled or
  unsupported responses, never after a stream starts. Isolate late events by request/thread identity
  and persist only completed chat messages.
- Routes are `/` (chat), `/cellar`, and `/taste-profile`. Layout and navigation live in
  `src/app/layout.tsx` and `src/components/Navigation.tsx`.
- Shared UI is in `src/components/`; domain components are grouped under `cellar/` and
  `taste-profile/`. Recharts wrappers are in `src/components/charts/`.

## Verification and Environment

- From the repository root: `make frontend-test` and `make frontend-build`.
- From `frontend/`: `npm run lint`, `npm test`, `npm run test:watch`, and `npm run test:coverage`.
- Tests use Vitest and React Testing Library; inspect tests beside the affected component.
- `NEXT_PUBLIC_API_URL` defaults to `http://localhost:8000/api`. Sensitive local environment files
  remain subject to the root approval rules.

<!-- BEGIN:nextjs-agent-rules -->
# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` before writing any code. Heed deprecation notices.
<!-- END:nextjs-agent-rules -->
