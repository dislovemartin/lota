cmake_minimum_required(VERSION 3.14)
project(cpp_module LANGUAGES CXX)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable testing
enable_testing()

# Include FetchContent module
include(FetchContent)

# Declare Catch2 dependency
FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.3.2 # Ensure this tag exists; adjust if necessary
)

# Make Catch2 available
FetchContent_MakeAvailable(catch2)

# Add the library
add_library(cpp_anomaly_detector src/cpp_anomaly_detector.cpp)

# Specify include directories for the library
target_include_directories(cpp_anomaly_detector PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

# Add test executable
add_executable(test_cpp_anomaly_detector tests/test_cpp_anomaly_detector.cpp)

# Link Catch2 and the library to the test executable
target_link_libraries(test_cpp_anomaly_detector PRIVATE Catch2::Catch2 cpp_anomaly_detector)

# Register the test
add_test(NAME cpp_anomaly_detector_tests COMMAND test_cpp_anomaly_detector)
