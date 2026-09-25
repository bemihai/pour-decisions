# Ollama Cloud Model Configuration

> **Project version**: 0.9.0 — last verified 2026-09-25.

Pour Decisions uses the direct Ollama Cloud API for application answers, descriptions, full-eval execution, and eval judging. Set `OLLAMA_API_KEY` privately in `.env`; do not put it in `app_config.yml` or commit it.

The application model is configured in `app_config.yml`:

```yaml
model:
  provider: ollama
  name: gemma4:31b
  base_url: https://ollama.com
  timeout_seconds: 60
```

The `eval.execution_model` and `eval.ragas.evaluator_model` slots are separate so judge settings can be reviewed independently. Both use direct Cloud access; the judge timeout defaults to 120 seconds. Change a model name only after checking that the selected Cloud model supports the required tool-calling and structured-output behavior. Cloud availability, rate limits, and pricing can change, so verify them with Ollama before a large run.

Local Ollama daemon URLs, local model pulls, hybrid planner/generator selection, and other-provider fallback settings are unsupported. Legacy settings fail explicitly. Retrieval still uses local embeddings and reranking; those are separate from generative inference.

A retrieval-only eval does not call the generative model. Full eval does, including separately configured judge calls. Plan bounded runs and report actual billing as unknown unless provider billing data is available.
