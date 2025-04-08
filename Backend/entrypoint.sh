#!/bin/bash
set -e

# Run the ChromaDB fix script
echo "Running ChromaDB fix script..."
python -m Backend.fix_chromadb

# Start the FastAPI application
echo "Starting FastAPI application..."
exec python -m uvicorn Backend.api:app --host 0.0.0.0 --port 8000