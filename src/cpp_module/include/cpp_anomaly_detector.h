#ifndef CPP_ANOMALY_DETECTOR_H
#define CPP_ANOMALY_DETECTOR_H

#include <string>
#include <vector>

class CppAnomalyDetector {
public:
    std::vector<std::string> detect(const std::string& data);
};

#endif // CPP_ANOMALY_DETECTOR_H
