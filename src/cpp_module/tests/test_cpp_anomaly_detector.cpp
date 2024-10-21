#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include "cpp_anomaly_detector.h"

TEST_CASE("Anomaly Detection Test", "[anomaly]") {
    CppAnomalyDetector detector;
    std::string data = "This is a test with error and warning.";
    std::vector<std::string> anomalies = detector.detect(data);

    REQUIRE(anomalies.size() == 2);
    REQUIRE(anomalies[0] == "Error detected in data.");
    REQUIRE(anomalies[1] == "Warning detected in data.");
}
