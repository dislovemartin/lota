#!/bin/bash
# install_dependencies.sh (Dependency Installation and Build Script)

set -euo pipefail
IFS=$'
	'

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r src/predictive_analytics/data_processing/requirements.txt
echo "Python dependencies installed."

echo "Installing Go dependencies..."
cd src/personalization_engine/real_time_personalization/go_service
go mod tidy
echo "Go dependencies installed."

echo "Installing Rust dependencies and building..."
cd ../../../../cybersecurity_ai/anomaly_detection/rust_service
cargo build --release
echo "Rust dependencies installed and project built."

echo "Installing C++ dependencies and building..."
cd ../../../../../cpp_module
mkdir -p build
cd build
cmake ..
make
echo "C++ dependencies installed and project built."

echo "All dependencies installed and projects built successfully."

