#!/bin/bash
# test_rust_anomaly_detector.sh (Unit Test Script for Rust Anomaly Detector)

set -euo pipefail
IFS=$'
	'

echo "Running Rust Anomaly Detector Tests..."

# Navigate to Rust project
cd src/cybersecurity_ai/anomaly_detection/rust_service

# Run tests
cargo test

echo "Rust Anomaly Detector Tests Passed."

