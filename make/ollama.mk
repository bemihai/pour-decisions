# ============================================================================
# ============================================================================

.PHONY: ollama-up
ollama-up:  ## Start Ollama server in the background (skipped if disabled or not installed)
	@if [ "$${OLLAMA_ENABLED:-true}" = "false" ] || [ "$${MODEL_PROVIDER:-local}" = "google" ]; then \
		echo "Skipping Ollama startup: local Ollama provider is disabled."; \
	elif ! command -v ollama > /dev/null 2>&1; then \
		echo "Skipping Ollama startup: Ollama CLI is not installed."; \
	else \
		echo "Starting Ollama server..."; \
		pgrep -x ollama > /dev/null 2>&1 && echo "Ollama already running." || (ollama serve > /tmp/ollama.log 2>&1 & echo "Ollama started (PID $$!)"); \
		for i in 1 2 3 4 5; do \
			curl -s http://localhost:11434/api/tags > /dev/null 2>&1 && break || sleep 2; \
		done; \
		curl -s http://localhost:11434/api/tags > /dev/null 2>&1 \
			&& echo "Ollama ready on http://localhost:11434" \
			|| echo "ERROR: Ollama failed to start. Check /tmp/ollama.log"; \
	fi

.PHONY: ollama-pull
ollama-pull:  ## Pull the configured model (see OLLAMA_MODEL in .env)
	@echo "Pulling $(OLLAMA_MODEL)..."
	ollama pull $(OLLAMA_MODEL)
	@echo "Model ready."

.PHONY: ollama-status
ollama-status:  ## Show running Ollama models and server info
	@echo "Ollama status:"
	@curl -s http://localhost:11434/api/tags 2>/dev/null \
		| python3 -c "import sys,json; d=json.load(sys.stdin); [print('  -', m['name'], m['details'].get('parameter_size',''), m['details'].get('quantization_level','')) for m in d.get('models',[])]" \
		|| echo "  Ollama not running. Run 'make ollama-up'."

.PHONY: ollama-models
ollama-models:  ## List all available models
	@echo "Downloaded Ollama models:"
	@ollama list 2>/dev/null || echo "  Ollama not running. Run 'make ollama-up'."

.PHONY: logs-ollama
logs-ollama:
	@echo "Viewing Ollama logs (Ctrl+C to exit)..."
	@docker compose logs -f --tail=100 ollama

.PHONY: shell-ollama
shell-ollama:
	@echo "Accessing Ollama container shell..."
	@docker compose exec ollama /bin/bash
