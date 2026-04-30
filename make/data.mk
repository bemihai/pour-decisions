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

