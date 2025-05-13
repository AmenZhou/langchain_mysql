#!/bin/bash

# IMPORTANT: Always run tests using this script to ensure proper test environment setup
# This script handles MySQL container initialization and cleanup automatically

# Exit on error
set -e

# Define the project root and test compose file
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.test.yml"

echo "Running tests in Docker container..."

# Start MySQL container
docker-compose -f "$COMPOSE_FILE" up -d mysql

# Wait for MySQL to be ready
echo "Waiting for MySQL to be ready..."
until docker-compose -f "$COMPOSE_FILE" exec -T mysql mysqladmin ping -h localhost -u test_user -ptestpassword --silent; do
    sleep 1
done
echo "MySQL is ready!"

# Build and run test container
docker-compose -f "$COMPOSE_FILE" build backend
docker-compose -f "$COMPOSE_FILE" up --exit-code-from backend backend

# Run pytest in the backend container
docker-compose -f "$COMPOSE_FILE" exec -T langchain_app pytest --maxfail=3 --disable-warnings --tb=short

# Cleanup
docker-compose -f "$COMPOSE_FILE" down --remove-orphans 
