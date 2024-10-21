cmake_minimum_required(VERSION 3.10)
project(CppAnomalyDetector)

set(CMAKE_CXX_STANDARD 17)

add_library(cpp_anomaly_detector STATIC src/cpp_anomaly_detector.cpp)

# Add include directories
target_include_directories(cpp_anomaly_detector PUBLIC include)

# Add executable for testing
add_executable(test_cpp_anomaly_detector tests/test_cpp_anomaly_detector.cpp)

# Link libraries
target_link_libraries(test_cpp_anomaly_detector cpp_anomaly_detector)

