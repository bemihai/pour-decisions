# ============================================================================
# ChromaDB Commands
# ============================================================================

.PHONY: chroma-upload
chroma-upload:
	@echo "Populating ChromaDB with wine knowledge (incremental mode)..."
	@PYTHONPATH=$(shell pwd) uv run python -m src.chroma.load_data
	@echo "ChromaDB indexing complete"

.PHONY: chroma-reindex
chroma-reindex:
	@echo "Force reindexing ChromaDB and rebuilding synchronized BM25..."
	@PYTHONPATH=$(shell pwd) uv run python -m src.chroma.load_data --force
	@echo "ChromaDB and BM25 reindexing complete"

.PHONY: chroma-status
chroma-status:
	@echo "Checking ChromaDB index status..."
	@PYTHONPATH=$(shell pwd) uv run python -m src.chroma.load_data --status

.PHONY: chroma-stats
chroma-stats:
	@echo "Getting ChromaDB collection statistics..."
	@PYTHONPATH=$(shell pwd) uv run python -m src.chroma.stats

CORPUS_STATS_OUTPUT ?= eval-results/m3_gate0_corpus_$(shell date +%Y%m%d).json
QUALITY_CALIBRATION_OUTPUT ?= eval-results/m3a_quality_calibration_$(shell date +%Y%m%d).json

.PHONY: chroma-stats-exact
chroma-stats-exact:
	@echo "Capturing exact ChromaDB corpus statistics..."
	@PYTHONPATH=$(shell pwd) uv run python -m src.chroma.stats --exact --output "$(CORPUS_STATS_OUTPUT)"
	@echo "Exact corpus artifact: $(CORPUS_STATS_OUTPUT)"

.PHONY: chroma-quality-calibration
chroma-quality-calibration:
	@echo "Generating chunk-quality calibration diagnostics..."
	@PYTHONPATH=$(shell pwd) uv run python -m src.eval.scripts.chunk_quality_calibration --output "$(QUALITY_CALIBRATION_OUTPUT)"
	@echo "Quality calibration artifact: $(QUALITY_CALIBRATION_OUTPUT)"

.PHONY: chroma-up
chroma-up:
	@echo "Starting ChromaDB container for local development..."
	@docker compose up chromadb -d --remove-orphans
	@echo "ChromaDB starting on http://localhost:8100"
	@$(MAKE) chroma-wait

# Internal target: poll until ChromaDB container reports healthy (max 60 s).
# Called by chroma-up so every dependent target benefits automatically.
.PHONY: chroma-wait
chroma-wait:
	@echo "Waiting for ChromaDB to be healthy..."
	@for i in $$(seq 1 30); do \
		if docker ps --filter "name=pour_decisions_chromadb" --filter "health=healthy" | grep -q chromadb; then \
			echo "ChromaDB is healthy and ready!"; \
			exit 0; \
		fi; \
		echo "  still waiting... ($$i/30, $$(( $$i * 2 ))s elapsed)"; \
		sleep 2; \
	done; \
	echo "ERROR: ChromaDB did not become healthy after 60 s. Check 'make chroma-health'."; \
	exit 1

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
