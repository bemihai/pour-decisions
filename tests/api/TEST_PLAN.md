# API Integration Test Plan

> **Project version**: 0.13.0 — last verified 2026-10-26.

This is the maintained manual smoke plan for the live FastAPI surface. Automated route, schema,
error, async-runtime, streaming, memory, tracing, and tool-registry coverage lives beside it in
`tests/api/`. Use the generated OpenAPI document as the request/response contract; this plan avoids
duplicating every schema field.

Live model, Tavily, CellarTracker, and populated-database checks are optional and must be reported
as unverified when their prerequisites are unavailable. Never expose credential values in test
output.

## Start the application

```bash
make chroma-up
make api
```

The standard backend port is `8000`:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/openapi.json
```

Interactive documentation is at <http://localhost:8000/docs>. A healthy response has `status: ok`
and Boolean availability for `cloud_model`, `intelligent_agent`, `retriever`, and `reranker`.

## Route inventory

Confirm the OpenAPI document contains this complete current surface:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Process and resource availability |
| `POST` | `/api/chat/` | Blocking Intelligent or RAG-only chat |
| `POST` | `/api/chat/stream` | Intelligent-mode SSE progress and final response |
| `GET` | `/api/chat/initial-message` | New-chat welcome message |
| `DELETE` | `/api/chat/threads/{thread_id}` | Idempotent conversation-thread deletion |
| `GET` | `/api/cellar/inventory` | Filtered and sorted owned inventory |
| `GET` | `/api/cellar/filters` | Full inventory filter options |
| `GET` | `/api/cellar/stats` | Cellar overview, drinking, and value statistics |
| `GET` | `/api/cellar/charts` | Chart-ready cellar aggregates |
| `GET` | `/api/cellar/drink-next` | Ready-to-drink wines grouped by type |
| `POST` | `/api/cellar/sync` | Explicit CellarTracker sync |
| `GET` | `/api/taste-profile/preferences` | Declared-preference list |
| `POST` | `/api/taste-profile/preferences` | Create a declared preference |
| `GET` | `/api/taste-profile/preferences/options` | Canonical preference options |
| `PATCH` | `/api/taste-profile/preferences/{preference_id}` | Version-checked preference update |
| `DELETE` | `/api/taste-profile/preferences/{preference_id}` | Version-checked preference delete |
| `POST` | `/api/taste-profile/preferences/reset` | Count-confirmed profile reset |
| `GET` | `/api/taste-profile/overview` | Taste overview |
| `GET` | `/api/taste-profile/rating-distribution` | Rating buckets |
| `GET` | `/api/taste-profile/wine-types` | Wine-type statistics |
| `GET` | `/api/taste-profile/varietals` | Ranked varietals |
| `GET` | `/api/taste-profile/producers` | Ranked producers |
| `GET` | `/api/taste-profile/regions` | Ranked regions |
| `GET` | `/api/taste-profile/countries` | Ranked countries |
| `GET` | `/api/taste-profile/vintages` | Vintage statistics |
| `GET` | `/api/taste-profile/appellations` | Appellation statistics |
| `GET` | `/api/taste-profile/rating-trends` | Monthly rating trend |
| `GET` | `/api/taste-profile/consumed` | Filtered tasting history |
| `GET` | `/api/wines/{wine_id}` | Wine, grape, producer, region, and bottle detail |
| `POST` | `/api/wines/{wine_id}/description` | Explicit wine-description generation |
| `POST` | `/api/wines/{wine_id}/producer-description` | Explicit producer-description generation |
| `POST` | `/api/grapes/{grape_id}/description` | Explicit grape-description generation |
| `GET` | `/api/tools` | Tool catalogue, readiness, and startup selection |

## Read-only smoke checks

Run these before any mutating or external-service check:

```bash
curl http://localhost:8000/api/chat/initial-message
curl http://localhost:8000/api/cellar/inventory
curl http://localhost:8000/api/cellar/filters
curl http://localhost:8000/api/cellar/stats
curl http://localhost:8000/api/cellar/charts
curl "http://localhost:8000/api/cellar/drink-next?limit=5"
curl http://localhost:8000/api/taste-profile/overview
curl http://localhost:8000/api/taste-profile/preferences
curl http://localhost:8000/api/taste-profile/preferences/options
curl "http://localhost:8000/api/taste-profile/consumed?limit=5"
curl http://localhost:8000/api/tools
```

Verify that responses match OpenAPI, empty databases return valid empty structures, filter options
come from the full data set, and errors do not expose tracebacks, credentials, hosts, or local paths.
For `/api/tools`, verify `available + unavailable == total`, every row contains the public metadata
contract, and `selected_for_agent` reflects the immutable startup snapshot. Do not hard-code the
catalogue size; the authoritative registry is `src/agents/tools/catalog.py`.

With known IDs from inventory detail, also check:

```bash
curl http://localhost:8000/api/wines/1
curl -i http://localhost:8000/api/wines/999999
```

The known record must match `WineDetailResponse`; the missing record must return `404` without
internal details.

## Chat and memory

Use RAG-only mode for a retrieval smoke check when the book index is populated:

```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H 'Content-Type: application/json' \
  -d '{"message":"What grapes are used in Burgundy?","agent_mode":"rag_only"}'
```

Use Intelligent mode only when the Cloud model loaded successfully:

```bash
curl -X POST http://localhost:8000/api/chat/ \
  -H 'Content-Type: application/json' \
  -d '{"message":"What wines are ready to drink in my cellar?","agent_mode":"intelligent"}'
```

Verify the echoed mode, nonempty answer, request ID, source shapes, and bounded friendly errors.
When session memory is enabled, send a UUID `thread_id`, exercise the OpenAPI-documented
`thread_action` values, then delete the thread. Deletion returns `204` and is idempotent:

```bash
curl -i -X DELETE http://localhost:8000/api/chat/threads/00000000-0000-4000-8000-000000000001
```

For SSE, post the same Intelligent request to `/api/chat/stream`. Verify `text/event-stream`, only
allowlisted progress fields, one authoritative final response, bounded heartbeat comments, and no
automatic replay after a post-start failure. RAG-only streaming and disabled streaming must fail
before SSE headers with their stable pre-execution response.

## Mutating and external checks

These checks change local state or call approved configured services. Run them only with suitable
test data, backups, and credentials:

- Preference create, patch, delete, and reset: verify strict variant validation, optimistic version
  conflicts, duplicate handling, reset count confirmation, and preservation on failed requests.
- Wine, producer, and grape description generation: verify explicit regeneration, supported
  nonempty text, cache persistence, `404` for missing entities, and bounded `502` failures. Record
  whether RAG and web search were enabled.
- CellarTracker sync: verify missing credentials return `400`; with valid credentials, reconcile the
  reported counts against the test cellar and record the external call.

Do not perform these against the user's real cellar merely to complete a documentation check.

## Automated verification

```bash
make test-fast TEST_PATH=tests/api
```

The suite covers application lifespan ownership, every route family, typed contracts, validation,
safe errors, async cancellation, conversation memory, SSE semantics, tracing, and registry rollout.
Report live checks separately; automated tests do not prove external service quality or real-data
correctness.
