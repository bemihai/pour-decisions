# Check if .env file exists
ifneq (,$(wildcard .env))
    include .env
    export
endif

# Default shell
SHELL := /bin/bash

# Default goal
.DEFAULT_GOAL := help

# Configuration variables
CELLAR_DB_PATH ?= cellar-data/wine_cellar.db
CELLAR_BACKUP_DIR ?= backups/wine_cellar

.PHONY: help
help:
	@echo "Pour Decisions Wine RAG - Available Commands"
	@echo ""
	@echo "Docker Compose Commands:"
	@echo "  up              - Start all services (app + ChromaDB)"
	@echo "  down            - Stop all services"
	@echo "  restart         - Restart all services"
	@echo "  logs            - View all service logs"
	@echo "  logs-app        - View app logs only"
	@echo "  logs-chroma     - View ChromaDB logs only"
	@echo "  status          - Check service status"
	@echo "  build           - Rebuild Docker images"
	@echo "  rebuild         - Stop, rebuild, and start services"
	@echo "  shell-app       - Access app container shell"
	@echo "  shell-chroma    - Access ChromaDB container shell"
	@echo ""
	@echo "Development Commands:"
	@echo "  install         - Install Python dependencies with uv"
	@echo "  run             - Run app locally with ChromaDB (PYTHONPATH configured)"
	@echo "  chroma-upload   - Populate ChromaDB with wine knowledge (incremental)"
	@echo "  chroma-reindex  - Force reindex all files in ChromaDB"
	@echo "  chroma-status   - Show index status (files and chunks)"
	@echo "  chroma-stats    - Show ChromaDB collection statistics"
	@echo "  chroma-up       - Start only ChromaDB (for local development)"
	@echo "  chroma-down     - Stop ChromaDB container"
	@echo "  chroma-health   - Check ChromaDB container health status"
	@echo "  chroma-reset    - Reset ChromaDB (stop, remove container, clear data)"
	@echo "  chroma-backup   - Backup ChromaDB data directory"
	@echo "  chroma-restore  - Restore ChromaDB from backup (BACKUP_FILE=path/to/backup.tar.gz)"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test            - Run all tests with coverage report"
	@echo "  test-unit       - Run tests with 80% coverage threshold"
	@echo "  test-fast       - Quick test run (no coverage, stop at first failure)"
	@echo "  test-watch      - Watch mode for continuous testing"
	@echo "  test-coverage   - Open HTML coverage report in browser"
	@echo ""
	@echo "Wine Cellar Database Commands:"
	@echo "  cellar-init     - Initialize wine cellar database"
	@echo "  cellar-info     - Show wine cellar database info"
	@echo "  cellar-backup   - Backup wine cellar database"
	@echo "  cellar-restore  - Restore from backup (BACKUP_FILE=path/to/backup.db)"
	@echo ""
	@echo "Data Import Commands:"
	@echo "  import-vivino   - Import Vivino CSV data"
	@echo "  import-ct       - Import from CellarTracker API"
	@echo "  sync            - Sync all sources (with auto-backup)"
	@echo ""
	@echo "Web Search Commands:"
	@echo "  web-cache-clear - Clear the web search result cache"

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
	@echo "Access app at: http://localhost:8501"

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
	@echo "Starting app locally..."
	@if ! docker ps --filter "name=pour_decisions_chromadb" --filter "status=running" --filter "health=healthy" | grep -q chromadb; then \
		echo "ChromaDB container not healthy. Starting/restarting..."; \
		$(MAKE) chroma-up; \
		echo "Waiting for ChromaDB to be healthy..."; \
		for i in 1 2 3 4 5 6 7 8 9 10; do \
			if docker ps --filter "name=pour_decisions_chromadb" --filter "health=healthy" | grep -q chromadb; then \
				echo "ChromaDB is healthy and ready!"; \
				break; \
			fi; \
			echo "Waiting for ChromaDB... ($$i/10)"; \
			sleep 2; \
		done; \
	fi
	@echo "Starting Streamlit app..."
	@PYTHONPATH=$(shell pwd) streamlit run src/ui/app.py

.PHONY: api
api:  ## Start FastAPI backend API (port 8000)
	@echo "Starting FastAPI backend API on :8000..."
	@if ! docker ps --filter "name=pour_decisions_chromadb" --filter "status=running" --filter "health=healthy" | grep -q chromadb; then \
		echo "ChromaDB container not healthy. Starting/restarting..."; \
		$(MAKE) chroma-up; \
		echo "Waiting for ChromaDB to be healthy..."; \
		for i in 1 2 3 4 5 6 7 8 9 10; do \
			if docker ps --filter "name=pour_decisions_chromadb" --filter "health=healthy" | grep -q chromadb; then \
				echo "ChromaDB is healthy and ready!"; \
				break; \
			fi; \
			echo "Waiting for ChromaDB... ($$i/10)"; \
			sleep 2; \
		done; \
	fi
	@PYTHONPATH=$(shell pwd) uvicorn src.api.main:app --reload --port 8000

.PHONY: frontend
frontend:  ## Start Next.js dev server (port 3000)
	@echo "Starting Next.js dev server on :3000..."
	@cd frontend && npm run dev

.PHONY: dev-full
dev-full:  ## Start ChromaDB + FastAPI + Next.js (all services for local dev)
	@echo "Starting all dev services (ChromaDB on :8100, FastAPI on :8000, Next.js on :3000)..."
	@$(MAKE) chroma-up
	@echo "Starting FastAPI and Next.js (Ctrl+C to stop all)..."
	@trap 'kill 0' EXIT; \
		PYTHONPATH=$(shell pwd) uvicorn src.api.main:app --reload --port 8000 & \
		(cd frontend && npm run dev) & \
		wait

.PHONY: frontend-build
frontend-build:  ## Production build of the Next.js app
	@echo "Building Next.js app for production..."
	@cd frontend && npm run build
	@echo "Build complete"

.PHONY: frontend-test
frontend-test:  ## Run frontend tests
	@echo "Running frontend tests..."
	@cd frontend && npm test

.PHONY: chroma-upload
chroma-upload:
	@echo "Populating ChromaDB with wine knowledge (incremental mode)..."
	@PYTHONPATH=$(shell pwd) python3 -m src.chroma.load_data
	@echo "ChromaDB indexing complete"

.PHONY: chroma-reindex
chroma-reindex:
	@echo "Force reindexing all files to ChromaDB..."
	@PYTHONPATH=$(shell pwd) python3 -m src.chroma.load_data --force
	@echo "ChromaDB reindexing complete"

.PHONY: chroma-status
chroma-status:
	@echo "Checking ChromaDB index status..."
	@PYTHONPATH=$(shell pwd) python3 -m src.chroma.load_data --status

.PHONY: chroma-stats
chroma-stats:
	@echo "Getting ChromaDB collection statistics..."
	@PYTHONPATH=$(shell pwd) python3 -m src.chroma.stats

.PHONY: chroma-up
chroma-up:
	@echo "Starting ChromaDB container for local development..."
	@docker compose up chromadb -d --remove-orphans
	@echo "ChromaDB starting on http://localhost:8100"
	@echo "Waiting for health check..."
	@sleep 3

.PHONY: chroma-down
chroma-down:
	@echo "Stopping ChromaDB container..."
	@docker compose stop chromadb
	@echo "ChromaDB stopped"

.PHONY: chroma-health
chroma-health:
	@echo "Checking ChromaDB health status..."
	@if docker ps -a --filter "name=pour_decisions_chromadb" --format "{{.Names}}" | grep -q chromadb; then \
		echo "Container Status:"; \
		docker ps -a --filter "name=pour_decisions_chromadb" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"; \
		echo ""; \
		if docker ps --filter "name=pour_decisions_chromadb" --filter "status=running" | grep -q chromadb; then \
			echo "Health Status: Running"; \
			echo "Testing connection to http://localhost:8100..."; \
			curl -s http://localhost:8100/api/v2/heartbeat > /dev/null && echo "Connection: OK" || echo "Connection: FAILED"; \
		else \
			echo "Health Status: Not Running"; \
		fi; \
	else \
		echo "ChromaDB container does not exist"; \
		echo "Run 'make chroma-up' to start it"; \
	fi

.PHONY: chroma-reset
chroma-reset:
	@echo "WARNING: This will completely reset ChromaDB (stop, remove container, clear data)"
	@read -p "Type 'yes' to continue: " confirm && [ "$$confirm" = "yes" ] || (echo "Reset cancelled" && exit 1)
	@echo "Stopping and removing ChromaDB container..."
	@docker compose stop chromadb 2>/dev/null || true
	@docker compose rm -f chromadb 2>/dev/null || true
	@echo "Clearing ChromaDB data..."
	@rm -rf chroma-data
	@mkdir -p chroma-data
	@echo "ChromaDB reset complete!"
	@echo "To restore from backup: make chroma-restore BACKUP_FILE=backups/chroma/chroma-backup-YYYYMMDD-HHMMSS.tar.gz"
	@echo "To populate fresh data: make chroma-upload"

.PHONY: chroma-backup
chroma-backup:
	@echo "Creating backup of ChromaDB data..."
	@if [ ! -d "chroma-data" ]; then \
		echo "Error: chroma-data directory not found"; exit 1; \
	fi
	@mkdir -p backups/chroma
	@tar -czf backups/chroma/chroma-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz -C chroma-data .
	@echo "Backup created in backups/chroma/"
	@ls -lh backups/chroma/ | tail -5

.PHONY: chroma-restore
chroma-restore:
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "Available ChromaDB backups:"; \
		ls -lht backups/chroma/ 2>/dev/null || (echo "No backups found in backups/chroma/" && exit 1); \
		echo ""; \
		echo "Usage: make chroma-restore BACKUP_FILE=backups/chroma/chroma-backup-YYYYMMDD-HHMMSS.tar.gz"; \
		exit 0; \
	fi
	@if [ ! -f "$(BACKUP_FILE)" ]; then \
		echo "Error: Backup file not found: $(BACKUP_FILE)"; \
		exit 1; \
	fi
	@echo "WARNING: This will overwrite existing ChromaDB data!"
	@echo "Backup file: $(BACKUP_FILE)"
	@read -p "Type 'yes' to continue: " confirm && [ "$$confirm" = "yes" ] || (echo "Restore cancelled" && exit 1)
	@echo "Stopping ChromaDB container..."
	@docker compose stop chromadb 2>/dev/null || true
	@docker compose rm -f chromadb 2>/dev/null || true
	@echo "Clearing existing data..."
	@rm -rf chroma-data
	@mkdir -p chroma-data
	@echo "Restoring from $(BACKUP_FILE)..."
	@tar -xzf $(BACKUP_FILE) -C chroma-data
	@echo "ChromaDB data restored successfully!"
	@echo "Run 'make chroma-up' to start ChromaDB with restored data"

# ============================================================================
# Wine Cellar Database Commands
# ============================================================================

.PHONY: cellar-init
cellar-init:
	@echo "Initializing wine cellar database..."
	@python3 -c "from src.database import initialize_database; success = initialize_database('$(CELLAR_DB_PATH)'); exit(0 if success else 1)"
	@echo "Wine cellar database initialized at $(CELLAR_DB_PATH)"

.PHONY: cellar-info
cellar-info:
	@echo "Wine Cellar Database Information:"
	@echo "=================================="
	@if [ -f "$(CELLAR_DB_PATH)" ]; then \
		echo "Status: Initialized"; \
		echo "Location: $(CELLAR_DB_PATH)"; \
		echo "Size: $$(du -h $(CELLAR_DB_PATH) | cut -f1)"; \
		echo ""; \
		echo "Statistics:"; \
		sqlite3 $(CELLAR_DB_PATH) "SELECT 'Producers: ' || COUNT(*) FROM producers UNION ALL SELECT 'Regions: ' || COUNT(*) FROM regions UNION ALL SELECT 'Wines: ' || COUNT(*) FROM wines UNION ALL SELECT 'Bottles: ' || COUNT(*) FROM bottles;" 2>/dev/null || echo "N/A"; \
	else \
		echo "Status: Not initialized"; \
		echo "Run 'make cellar-init' to create the database"; \
	fi

.PHONY: cellar-backup
cellar-backup:
	@echo "Creating backup of wine cellar database..."
	@if [ ! -f "$(CELLAR_DB_PATH)" ]; then \
		echo "Error: Database not found at $(CELLAR_DB_PATH)"; exit 1; \
	fi
	@mkdir -p $(CELLAR_BACKUP_DIR)
	@cp $(CELLAR_DB_PATH) $(CELLAR_BACKUP_DIR)/wine_cellar_$(shell date +%Y%m%d_%H%M%S).db
	@echo "Backup created in $(CELLAR_BACKUP_DIR)/"
	@ls -lh $(CELLAR_BACKUP_DIR)/ | tail -5

.PHONY: cellar-restore
cellar-restore:
	@echo "Available backups:"
	@ls -lht $(CELLAR_BACKUP_DIR)/ 2>/dev/null || (echo "No backups directory found" && exit 1)
	@echo ""
	@echo "Usage: make cellar-restore BACKUP_FILE=$(CELLAR_BACKUP_DIR)/wine_cellar_YYYYMMDD_HHMMSS.db"
	@if [ -n "$(BACKUP_FILE)" ]; then \
		if [ ! -f "$(BACKUP_FILE)" ]; then \
			echo "Error: Backup file not found: $(BACKUP_FILE)"; exit 1; \
		fi; \
		echo "Restoring from $(BACKUP_FILE)..."; \
		mkdir -p cellar-data; \
		cp $(BACKUP_FILE) $(CELLAR_DB_PATH); \
		echo "Database restored from backup"; \
	fi

# ============================================================================
# Testing Commands
# ============================================================================

.PHONY: test
test:
	@echo "Running all unit tests with coverage..."
	@PYTHONPATH=$(shell pwd) pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

.PHONY: test-unit
test-unit:
	@echo "Running unit tests with coverage threshold (80%)..."
	@PYTHONPATH=$(shell pwd) pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html --cov-fail-under=80

.PHONY: test-fast
test-fast:
	@echo "Running tests quickly (no coverage, stop at first failure)..."
	@PYTHONPATH=$(shell pwd) pytest tests/ -v -x

.PHONY: test-watch
test-watch:
	@echo "Running tests in watch mode..."
	@PYTHONPATH=$(shell pwd) ptw tests/ -- -v --cov=src --cov-report=term-missing

.PHONY: test-coverage
test-coverage:
	@echo "Opening coverage report in browser..."
	@open htmlcov/index.html 2>/dev/null || xdg-open htmlcov/index.html 2>/dev/null || echo "Coverage report: htmlcov/index.html"

# ============================================================================
# Data Import Commands
# ============================================================================

.PHONY: import-vivino
import-vivino:
	@echo "Importing Vivino CSV data..."
	@if [ ! -f "$(CELLAR_DB_PATH)" ]; then \
		echo "Error: Database not initialized. Run 'make cellar-init' first."; exit 1; \
	fi
	@PYTHONPATH=$(shell pwd) python3 src/etl/import_vivino.py
	@echo "Import completed!"
	@$(MAKE) cellar-info

.PHONY: import-ct
import-ct:
	@echo "Importing from CellarTracker API..."
	@if [ ! -f "$(CELLAR_DB_PATH)" ]; then \
		echo "Error: Database not initialized. Run 'make cellar-init' first."; exit 1; \
	fi
	@if [ -z "$(CELLAR_TRACKER_USERNAME)" ] || [ -z "$(CELLAR_TRACKER_PASSWORD)" ]; then \
		echo "Error: CellarTracker credentials not set!"; \
		echo "Set CELLAR_TRACKER_USERNAME and CELLAR_TRACKER_PASSWORD in .env file"; exit 1; \
	fi
	@PYTHONPATH=$(shell pwd) python3 -m src.etl.import_cellartracker
	@echo "Import completed!"
	@$(MAKE) cellar-info

.PHONY: sync
sync:
	@echo "Syncing all wine data sources..."
	@$(MAKE) cellar-backup
	@echo ""
	@echo "Importing from CellarTracker..."
	@$(MAKE) import-ct
	@echo ""
	@echo "All sources synced!"

# ============================================================================
# Web Search Commands
# ============================================================================

.PHONY: web-cache-clear
web-cache-clear:
	@echo "Clearing web search cache..."
	@PYTHONPATH=$(shell pwd) python3 -c "\
from src.agents.tools.web_search_tools import WebSearchCache; \
from src.utils import get_config, get_project_root; \
cfg = get_config(); \
WebSearchCache(get_project_root() / cfg.web_search.cache.db_path).clear()"
	@echo "Web search cache cleared."

