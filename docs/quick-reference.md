# Pour Decisions - Quick Reference

## Common Commands

### Local Development
```bash
make install          # Install dependencies with uv
make run             # Run app locally (auto-starts ChromaDB if needed)
make chroma-health   # Check ChromaDB status
```

### Docker Deployment
```bash
make up              # Start all services (app + ChromaDB)
make down            # Stop all services
make logs            # View logs
make status          # Check service status
```

### ChromaDB Management
```bash
make chroma-up       # Start ChromaDB only
make chroma-down     # Stop ChromaDB
make chroma-health   # Check health status
make chroma-backup   # Backup ChromaDB data
make chroma-restore BACKUP_FILE=path/to/backup.tar.gz  # Restore from backup
make chroma-reset    # Complete reset (removes all data)
make populate-chroma # Populate with wine books
```

### Wine Cellar Database
```bash
make cellar-init     # Initialize database
make cellar-info     # Show database stats
make cellar-backup   # Backup database
make import-vivino   # Import Vivino CSV
make import-ct       # Import CellarTracker
make sync            # Sync all sources
```

### Testing
```bash
make test            # Run all tests with coverage report
make test-unit       # Run tests with 80% coverage threshold
make test-fast       # Quick test run (no coverage, stop at first failure)
make test-watch      # Watch mode for continuous testing
make test-coverage   # Open HTML coverage report in browser
```

## Port Configuration

| Service   | Local Port | Docker Port |
|-----------|-----------|-------------|
| Streamlit | 8501      | 8501        |
| ChromaDB  | 8000      | 8000        |

## Environment Variables

Required in `.env`:
```bash
GOOGLE_API_KEY=your_gemini_api_key
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
WINE_BOOKS_PATH=data/wine-books
```

Optional for CellarTracker import:
```bash
CELLAR_TRACKER_USERNAME=your_username
CELLAR_TRACKER_PASSWORD=your_password
```

## Directory Structure

```
chroma-data/          # ChromaDB persistent storage (mounted as volume)
cellar-data/          # Wine cellar SQLite database
  wine_cellar.db
backups/
  chroma/             # ChromaDB backups
  wine_cellar/        # Database backups
data/
  wine-books/         # Wine book PDFs for RAG
```

## Troubleshooting

### ChromaDB won't start
1. Check logs: `docker logs pour_decisions_chromadb`
2. Reset and restore: `make chroma-reset`, then `make chroma-restore BACKUP_FILE=...`

### Collection not found
1. Restore from backup: `make chroma-restore BACKUP_FILE=backups/chroma/chroma-backup-YYYYMMDD-HHMMSS.tar.gz`
2. Or repopulate: `make populate-chroma`

### Connection refused
1. Ensure ChromaDB is running: `make chroma-health`
2. Start if needed: `make chroma-up`
3. Wait for health check (up to 20 seconds)

For more details, see `TROUBLESHOOTING.md`

