#!/bin/bash
# Quick start script for Wine RAG Docker deployment

set -e

echo "Pour Decisions Wine RAG - Docker Setup"
echo "=========================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "WARNING: No .env file found."
    echo "Please create a .env file with your OLLAMA_API_KEY"
    echo ""
    read -p "Press Enter after you've created .env with your API key..."
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi

echo "Docker is running"

# Check if Ollama Cloud API key is set without printing the secret.
if ! grep -Eq '^OLLAMA_API_KEY=[^[:space:]]+' .env || grep -q '^OLLAMA_API_KEY=your_ollama_api_key_here' .env; then
    echo "ERROR: Please set your OLLAMA_API_KEY in .env file"
    exit 1
fi

echo "Environment variables configured"

# Build and start services
echo ""
echo "Building Docker images..."
docker compose build

echo ""
echo "Starting services..."
docker compose up -d

echo ""
echo "Waiting for services to be healthy..."
sleep 5

# Check ChromaDB health
if docker compose exec -T chromadb curl -f --max-time 5 http://localhost:8000/api/v1/heartbeat > /dev/null 2>&1; then
    echo "ChromaDB is healthy"
else
    echo "WARNING: ChromaDB may still be starting up..."
fi

echo ""
echo "=========================================="
echo "Deployment complete!"
echo ""
echo "Access your app at: http://localhost:3000"
echo ""
echo "The app uses the Ollama Cloud model configured in app_config.yml."
echo ""
echo "Useful commands:"
echo "  View logs:         make logs"
echo "  View API logs:     make logs-app"
echo "  Stop services:     make down"
echo "  Restart:           make restart"
echo "  View status:       make status"
echo ""
echo "Run 'make help' to see all available commands"
echo "=========================================="
