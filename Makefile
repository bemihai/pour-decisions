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

# Include sub-makefiles
include make/docker.mk
include make/chroma.mk
include make/dev.mk
include make/testing.mk
include make/cellar.mk
include make/data.mk
include make/ollama.mk

.PHONY: help
help:
	@echo "Pour Decisions Wine RAG - Available Commands"
	@echo ""
	@echo "Docker Compose Commands:"
	@echo "  up              - Start default services (api + frontend + ChromaDB)"
	@echo "  down            - Stop all services"
	@echo "  restart         - Restart all services"
	@echo "  logs            - View all service logs"
	@echo "  logs-app        - View app logs only"
	@echo "  logs-chroma     - View ChromaDB logs only"
	@echo "  logs-ollama     - View Ollama logs only"
	@echo "  status          - Check service status"
	@echo "  build           - Rebuild Docker images"
	@echo "  rebuild         - Stop, rebuild, and start services"
	@echo "  shell-app       - Access app container shell"
	@echo "  shell-chroma    - Access ChromaDB container shell"
	@echo "  shell-ollama    - Access Ollama container shell"
	@echo ""
	@echo "Development Commands:"
	@echo "  install         - Install Python dependencies with uv"
	@echo "  run             - Start production stack: ChromaDB + FastAPI + Next.js"
	@echo "  api             - Start FastAPI backend (port 8000, auto-starts ChromaDB)"
	@echo "  frontend        - Start Next.js dev server (port 3000)"
	@echo "  dev-full        - Start ChromaDB + FastAPI + Next.js (all at once)"
	@echo "  dev-stop        - Kill any lingering processes on :8000 and :3000"
	@echo "  frontend-build  - Production build of Next.js app"
	@echo "  frontend-test   - Run frontend tests"
	@echo "  chroma-upload   - Populate ChromaDB with wine knowledge (incremental)"
	@echo "  chroma-reindex  - Force reindex all files in ChromaDB"
	@echo "  chroma-status   - Show index status (files and chunks)"
	@echo "  chroma-stats    - Show sampled ChromaDB collection statistics"
	@echo "  chroma-stats-exact - Save exact configured-corpus JSON statistics"
	@echo "  chroma-up       - Start only ChromaDB (for local development)"
	@echo "  chroma-down     - Stop ChromaDB container"
	@echo "  chroma-health   - Check ChromaDB container health status"
	@echo "  chroma-reset    - Reset ChromaDB (stop, remove container, clear data)"
	@echo "  chroma-backup   - Backup ChromaDB data directory"
	@echo "  chroma-restore  - Restore ChromaDB from backup (BACKUP_FILE=path/to/backup.tar.gz)"
	@echo "  phoenix         - Start Phoenix observability dashboard (port 6006)"
	@echo "  phoenix-down    - Stop Phoenix dashboard"
	@echo "  ollama-up       - Start Ollama server (background)"
	@echo "  ollama-pull     - Pull the configured model (see OLLAMA_MODEL in .env)"
	@echo "  ollama-status   - Show running Ollama models and server info"
	@echo "  ollama-models   - List all available models"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test            - Run all tests (Python + frontend) with coverage report"
	@echo "  test-unit       - Run tests with 80% coverage threshold"
	@echo "  test-fast       - Quick test run (no coverage, stop at first failure)"
	@echo "  test-watch      - Watch mode for continuous testing"
	@echo "  test-coverage   - Open HTML coverage report in browser"
	@echo "  eval            - Run eval harness in retrieval-only mode (starts local Ollama)"
	@echo "  eval-full       - Run eval harness in full Ragas mode (starts local Ollama)"
	@echo "  eval-report     - Compare latest eval runs"
	@echo "  eval-validate   - Validate golden dataset against live cellar DB"
	@echo "  eval-curate     - Interactively assign ground_truth_chunk_ids in golden dataset"
	@echo "  eval-phoenix    - Run eval and push results to Phoenix"
	@echo "  eval-phoenix-full - Run full eval and push results to Phoenix"
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
	@echo ""
	@echo "Local LLM (Ollama) Commands:"
	@echo "  ollama-up       - Start Ollama server (background)"
	@echo "  ollama-pull     - Pull the configured model (see OLLAMA_MODEL in .env)"
	@echo "  ollama-status   - Show running Ollama models and server info"
	@echo "  ollama-models   - List all available models"
