#!/bin/bash

# Fast test runner using persistent volumes and BuildKit cache
# This script avoids reinstalling dependencies on every run

set -e

# Enable Docker BuildKit for cache mounts
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Define the project root and test compose file
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.test.yml"

echo "Running tests with persistent dependency cache..."

# Start MySQL container if not running
echo "Starting MySQL container..."
docker-compose -f "$COMPOSE_FILE" up -d mysql

# Wait for MySQL to be ready
echo "Waiting for MySQL to be ready..."
until docker-compose -f "$COMPOSE_FILE" exec -T mysql mysqladmin ping -h localhost -u test_user -ptestpassword --silent; do
    echo -n "."
    sleep 1
done
echo "MySQL is ready!"

# Check if we need to rebuild or can reuse existing container
REBUILD=""
if [[ "$1" == "--rebuild" ]]; then
    echo "Forcing rebuild of test container..."
    REBUILD="--build"
    docker-compose -f "$COMPOSE_FILE" build backend
fi

# Run tests using cached dependencies
echo "Running tests..."
docker-compose -f "$COMPOSE_FILE" run --rm $REBUILD backend

# Optional: Keep containers running for faster subsequent runs
if [[ "$1" != "--cleanup" ]]; then
    echo ""
    echo "Containers are left running for faster subsequent test runs."
    echo "Use '$0 --cleanup' to stop all containers when done."
    echo "Use '$0 --rebuild' to force rebuild the backend image."
else
    echo "Cleaning up containers..."
    docker-compose -f "$COMPOSE_FILE" down
fi 
