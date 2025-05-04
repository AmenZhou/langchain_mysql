#!/bin/bash

# Exit on error
set -e

# Print commands as they are executed
set -x

# Get the absolute path of the frontend directory
FRONTEND_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Check if the Docker image exists
if ! docker image inspect frontend-tests:latest >/dev/null 2>&1; then
    echo "Building Docker image for testing..."
    docker build -t frontend-tests -f "${FRONTEND_DIR}/Dockerfile.test" "${FRONTEND_DIR}"
else
    # Check if package.json has changed
    if [ "$(docker run --rm frontend-tests:latest sh -c 'cat package.json')" != "$(cat "${FRONTEND_DIR}/package.json")" ]; then
        echo "package.json has changed, rebuilding Docker image..."
        docker build -t frontend-tests -f "${FRONTEND_DIR}/Dockerfile.test" "${FRONTEND_DIR}"
    fi
fi

# Run tests in Docker container
echo "Running frontend tests in Docker..."
docker run --rm \
    -v "${FRONTEND_DIR}/src:/frontend/src" \
    -v "${FRONTEND_DIR}/public:/frontend/public" \
    -v "${FRONTEND_DIR}/jest.config.js:/frontend/jest.config.js" \
    -w /frontend \
    frontend-tests \
    npm test

# If we get here, tests passed
echo "All tests passed successfully!" 
