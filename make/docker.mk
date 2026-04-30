# ============================================================================
# Docker Compose Commands
# ============================================================================

.PHONY: up
up:
	@echo "Starting all services with Docker Compose..."
	@if [ ! -f .env ]; then \
		echo "WARNING: .env file not found. Create one with GOOGLE_API_KEY"; \
		exit 1; \
	fi
	@docker compose up -d
	@echo "Services started!"
	@echo "Access app at: http://localhost:3000"

.PHONY: down
down:
	@echo "Stopping all services..."
	@docker compose down
	@echo "Services stopped"

.PHONY: restart
restart:
	@echo "Restarting all services..."
	@docker compose restart
	@echo "Services restarted"

.PHONY: logs
logs:
	@echo "Viewing logs (Ctrl+C to exit)..."
	@docker compose logs -f --tail=100

.PHONY: logs-app
logs-app:
	@echo "Viewing app logs (Ctrl+C to exit)..."
	@docker compose logs -f --tail=100 app

.PHONY: logs-chroma
logs-chroma:
	@echo "Viewing ChromaDB logs (Ctrl+C to exit)..."
	@docker compose logs -f --tail=100 chromadb

.PHONY: status
status:
	@echo "Service Status:"
	@docker compose ps

.PHONY: build
build:
	@echo "Building Docker images..."
	@docker compose build --no-cache
	@echo "Build complete"

.PHONY: rebuild
rebuild: down build up

.PHONY: shell-app
shell-app:
	@echo "Accessing app container shell..."
	@docker compose exec app /bin/bash

.PHONY: shell-chroma
shell-chroma:
	@echo "Accessing ChromaDB container shell..."
	@docker compose exec chromadb /bin/bash

