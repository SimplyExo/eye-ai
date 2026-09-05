cmake_minimum_required(VERSION 3.23)

project(EyeAIVision_Main
    VERSION 1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Qt 6
find_package(Qt6 REQUIRED COMPONENTS Core Network)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
)

# Source code
file(GLOB_RECURSE SRC_FILES
    "src/*.cpp"
    "include/*.hpp"
)

# Executable
add_executable(${PROJECT_NAME} ${SRC_FILES})

# Link Qt 6 Core
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        Qt6::Core
        Qt6::Network
)

# Enable Qt's automatic MOC/UIC/RCC handling
set_target_properties(${PROJECT_NAME} PROPERTIES
    AUTOMOC ON
    AUTOUIC ON
    AUTORCC ON
)
