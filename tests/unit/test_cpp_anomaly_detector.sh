#!/bin/bash
# test_cpp_anomaly_detector.sh (Unit Test Script for C++ Anomaly Detector)

set -euo pipefail
IFS=$'
	'

echo "Compiling C++ Anomaly Detector Test..."

# Compile the C++ test
g++ -o test_cpp_anomaly_detector src/cpp_module/tests/test_cpp_anomaly_detector.cpp src/cpp_module/include/cpp_anomaly_detector.h src/cpp_module/src/cpp_anomaly_detector.cpp

echo "Running C++ Anomaly Detector Test..."
./test_cpp_anomaly_detector

echo "C++ Anomaly Detector Test Passed."

