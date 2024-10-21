#!/bin/bash
# test_runner.sh (Test Runner Script)

set -euo pipefail
IFS=$'
	'

echo "Running unit tests..."

# Run C++ Anomaly Detector Unit Test
bash tests/unit/test_cpp_anomaly_detector.sh

# Run Rust Anomaly Detector Unit Test
bash tests/unit/test_rust_anomaly_detector.sh

# Run Python Unit Tests
python3 -m unittest discover -s tests/unit

echo "Running integration tests..."
python3 -m unittest discover -s tests/integration

echo "Running end-to-end tests..."
python3 -m unittest discover -s tests/e2e

echo "All tests executed successfully."

