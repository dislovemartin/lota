#include "cpp_anomaly_detector.h"

std::vector<std::string> CppAnomalyDetector::detect(const std::string& data) {
    std::vector<std::string> anomalies;
    if (data.find("error") != std::string::npos) {
        anomalies.emplace_back("Error detected in data.");
    }
    if (data.find("warning") != std::string::npos) {
        anomalies.emplace_back("Warning detected in data.");
    }
    return anomalies;
}
