#!/bin/bash

# Get the absolute path of the backend directory
BACKEND_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Start the backend server using Docker
echo "Starting backend server..."
docker run --rm \
    -p 8000:8000 \
    -v "${BACKEND_DIR}:/app" \
    -e DATABASE_URL="mysql+pymysql://test_user:testpassword@host.docker.internal:3307/test_db" \
    -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
    -e PYTHONPATH="/app:/app/src:/app/backend/src" \
    --name langchain_mysql-langchain_app \
    langchain_mysql-langchain_app \
    uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 
