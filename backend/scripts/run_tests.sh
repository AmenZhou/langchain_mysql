#!/bin/bash

# IMPORTANT: Always run tests using this script to ensure proper test environment setup
# This script handles MySQL container initialization and cleanup automatically

# Exit on error
set -e

echo "Running tests in Docker container..."

# Start MySQL container
docker-compose -f docker-compose.test.yml up -d mysql

# Wait for MySQL to be ready
echo "Waiting for MySQL to be ready..."
until docker-compose -f docker-compose.test.yml exec -T mysql mysqladmin ping -h localhost -u test_user -ptestpassword --silent; do
    sleep 1
done
echo "MySQL is ready!"

# Build and run test container
docker-compose -f docker-compose.test.yml build backend
docker-compose -f docker-compose.test.yml up --exit-code-from backend backend

# Cleanup
docker-compose -f docker-compose.test.yml down 
