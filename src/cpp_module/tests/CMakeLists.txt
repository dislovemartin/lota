cmake_minimum_required(VERSION 3.10)
project(CppAnomalyDetectorTests)

# Add executable for tests
add_executable(test_cpp_anomaly_detector tests/test_cpp_anomaly_detector.cpp)

# Link libraries
target_link_libraries(test_cpp_anomaly_detector cpp_anomaly_detector)

