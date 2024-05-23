cmake_minimum_required(VERSION 3.10)
project(NeuralNet)

set(CMAKE_CXX_STANDARD 17)

# Add the include directory
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
file(GLOB SOURCES "src/*.cpp")

# Set the output directories for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})

# Set the build type to Debug
set(CMAKE_BUILD_TYPE Debug)

# Main executable
add_executable(NeuralNet ${SOURCES})

# Test executable
file(GLOB TEST_SOURCES "tests/*.cpp")
add_executable(NeuralNetTests ${TEST_SOURCES})
