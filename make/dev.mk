# ============================================================================
# Development Commands
# ============================================================================

.PHONY: install
install:
	@echo "Installing Python dependencies with uv..."
	@uv sync
	@echo "Dependencies installed"

.PHONY: run
run:
	@echo "Starting production stack (ChromaDB on :8100, FastAPI on :8000, Next.js on :3000)..."
	@$(MAKE) chroma-up
	@echo "Building Next.js production bundle..."
	@cd frontend && npm run build
	@echo "Starting FastAPI and Next.js (Ctrl+C to stop both)..."
	@cleanup() { \
		if [ -n "$$api_pid" ] && kill -0 $$api_pid 2>/dev/null; then \
			kill $$api_pid 2>/dev/null || true; \
			wait $$api_pid 2>/dev/null || true; \
		fi; \
		if [ -n "$$frontend_pid" ] && kill -0 $$frontend_pid 2>/dev/null; then \
			kill $$frontend_pid 2>/dev/null || true; \
			wait $$frontend_pid 2>/dev/null || true; \
		fi; \
	}; \
	trap cleanup EXIT INT TERM; \
	PYTHONPATH=$(shell pwd) uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & \
	api_pid=$$!; \
	(cd frontend && npm start) & \
	frontend_pid=$$!; \
	wait $$api_pid $$frontend_pid

.PHONY: api
api:
	@echo "Starting FastAPI backend API on :8000..."
	@if ! docker ps --filter "name=pour_decisions_chromadb" --filter "health=healthy" | grep -q chromadb; then \
		$(MAKE) chroma-up; \
	else \
		echo "ChromaDB already healthy."; \
	fi
	@PYTHONPATH=$(shell pwd) uvicorn src.api.main:app --reload --port 8000

.PHONY: frontend
frontend:
	@echo "Starting Next.js dev server on :3000..."
	@cd frontend && npm run dev

.PHONY: dev-stop
dev-stop:
	@echo "Stopping any existing dev processes on :8000 and :3000..."
	@lsof -ti:8000 | xargs kill -9 2>/dev/null && echo "  killed process(es) on :8000" || echo "  :8000 already free"
	@lsof -ti:3000 | xargs kill -9 2>/dev/null && echo "  killed process(es) on :3000" || echo "  :3000 already free"

.PHONY: dev-full
dev-full:
	@echo "Starting all dev services (ChromaDB on :8100, FastAPI on :8000, Next.js on :3000)..."
	@$(MAKE) dev-stop
	@if ! docker ps --filter "name=pour_decisions_chromadb" --filter "health=healthy" | grep -q chromadb; then \
		$(MAKE) chroma-up; \
	else \
		echo "ChromaDB already healthy."; \
	fi
	@echo "ChromaDB ready. Launching FastAPI and Next.js (Ctrl+C to stop all)..."
	@trap 'kill 0' EXIT; \
		PYTHONPATH=$(shell pwd) CHROMA_PORT=8100 uvicorn src.api.main:app --reload --port 8000 & \
		(cd frontend && npm run dev) & \
		wait

.PHONY: frontend-build
frontend-build:
	@echo "Building Next.js app for production..."
	@cd frontend && npm run build
	@echo "Build complete"

.PHONY: frontend-test
frontend-test:
	@echo "Running frontend tests..."
	@cd frontend && npm test

.PHONY: phoenix
phoenix:
	@echo "Starting Phoenix observability UI on :6006..."
	@docker compose up -d phoenix
	@echo "Phoenix UI available at http://localhost:6006"

.PHONY: phoenix-down
phoenix-down:
	@echo "Stopping Phoenix..."
	@docker compose stop phoenix

