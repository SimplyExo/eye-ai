cmake_minimum_required(VERSION 3.23)
project(EyeAIVision_Main)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_BINARY_DIR})

# Source code
file(GLOB_RECURSE SRC_FILES "src/*.cpp" "include/*.hpp")

# Executable
add_executable(${PROJECT_NAME} ${SRC_FILES})
