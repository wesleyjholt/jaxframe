#!/bin/bash
# Test runner script for jaxframe

echo "Running all tests with pytest..."
pytest

echo ""
echo "Running tests with coverage (if pytest-cov is installed)..."
pytest --cov=src/jaxframe --cov-report=term-missing 2>/dev/null || echo "pytest-cov not installed, skipping coverage"
