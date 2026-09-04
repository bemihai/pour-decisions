# Agent Runtime Guardrails

> **Project version:** 0.8.4 - last verified 2026-09-04.

The intelligent agent combines deterministic M9A request safeguards with M9B asynchronous tool
execution policy. These controls do not change the public chat request or response schema.

## Execution Scope

M9B applies only to selected tools executed through `WineAgent.ainvoke()` in intelligent standard
or hybrid mode. Synchronous `WineAgent.invoke()`, `stream()`, current eval execution, and the
`rag_only` pipeline retain their existing behavior without M9B admission, deadlines, retries, or
execution reporting.

The policy bounds how long the async graph waits for each tool call. It is not an end-to-end request
deadline: model calls, multiple tool rounds, and final generation can make a request take longer
than one tool deadline.

## Policy Flow

For each known tool call, the async `ToolNode` wrapper:

1. Selects metadata from the agent's immutable construction snapshot.
2. Starts one latency-class deadline before shared controller admission.
3. Acquires one app-worker-scoped async permit.
4. Executes the LangGraph handler through the existing M9A safe-error boundary.
5. Optionally retries one reviewed SQLite contention failure within the same deadline and permit.
6. Releases the permit and merges bounded outcomes into the request report.

Unknown tools receive no M9B policy and retain LangGraph invalid-tool handling. Disabling
`tool_execution` restores the existing M9A async wrapper unchanged.

```yaml
agents:
  guardrails:
    tool_execution:
      enabled: true
      max_concurrent_calls: 4
      timeout_seconds:
        fast: 10
        slow: 30
      retry:
        enabled: true
        max_attempts: 2
        delay_seconds: 0.1
        min_remaining_seconds: 1.0
        allowed_cost_classes:
          - free
```

`max_attempts: 2` means one initial attempt and at most one retry. A retry requires all of the
following: explicit idempotence, an allowed cost class, a structured SQLite `BUSY` or `LOCKED`
primary result code, and remaining time strictly greater than the fixed delay plus the minimum
remaining budget. Deadlines, upstream timeouts, generic failures, validation failures, missing
SQLite codes, non-idempotent tools, and disallowed cost classes never retry.

## Cancellation And Synchronous Work

Caller `CancelledError` and LangGraph `GraphBubbleUp` propagate unchanged. Coroutine deadline
cancellation is cooperative, and a cancellation-suppressing coroutine cannot return a late success
after the timeout context has expired.

All 18 built-in tools in the 0.8.4 baseline are synchronous and use LangChain's worker-thread
bridge on the async path. A deadline stops awaiting that bridge but cannot terminate its worker.
The admission permit is released when the wrapper returns, so repeated timeouts can leave continuing
workers beyond the configured admission limit. `tool_sync_timeout` records this exposure; it is not
a hard-cancellation or bounded-worker guarantee. Native async tools and explicitly owned bounded
executors remain M6B work.

## Internal Outcomes

One request-local `ToolExecutionReport` records only catalogue and policy fields. It never retains
arguments, results, exception objects, exception strings, environment identifiers, or secrets.

| Event | Meaning |
|-------|---------|
| `tool_deadline_exceeded` | The total deadline expired during admission, delay, or execution |
| `tool_sync_timeout` | Deadline cancellation may have left a synchronous worker running |
| `tool_retry_started` | Attempt 2 began |
| `tool_retry_succeeded` | Attempt 2 returned successfully |
| `tool_terminal_failure` | A non-deadline exception ended in an M9A-safe result |

The request span exposes aggregate counts only:

- `guardrail.tool.timeout.count`
- `guardrail.tool.sync_timeout.count`
- `guardrail.tool.retry.count`
- `guardrail.tool.retry_success.count`
- `guardrail.tool.terminal_failure.count`
- `guardrail.tool.concurrency.limit`

Existing `guardrail.tool_error.count` continues to count safe-error `ToolMessage` objects. M9B does
not rewrite completed child tool spans.

## Local Timing Evidence

Gate 0 measured two sequential synchronous invocations per representative tool in one local Python
process using `time.perf_counter()` on 2026-09-02:

| Tool | Cold | Warm |
|------|------|------|
| `get_cellar_statistics` | 0.119430 s | 0.043076 s |
| `search_wine_knowledge` | 16.484290 s | 3.102028 s |

The local RAG observation used the query "What makes Barolo distinctive?", five results, source
formatting disabled, a healthy ChromaDB service, and the 37,412-document BM25 index. These are
single-machine development observations, not production percentiles or timeout targets. They do
not justify changing the reviewed 10-second fast and 30-second slow defaults.
