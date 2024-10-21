// test_cpp_anomaly_detector.cpp (C++ Anomaly Detector Unit Tests)
#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include "cpp_anomaly_detector.h"

TEST_CASE("CppAnomalyDetector detects anomalies", "[AnomalyDetector]") {
    CppAnomalyDetector detector;
    std::vector<std::string> result = detector.detect("Test data");
    REQUIRE(result.size() == 1);
    REQUIRE(result[0] == "Anomaly Detected");
}

