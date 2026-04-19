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

