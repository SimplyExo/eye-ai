cmake_minimum_required(VERSION 3.23)

project(EyeAIVision_Main
    VERSION 1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(TARGET_ARCH "armv6" CACHE STRING "Target architecture: x86 or armv6")
set_property(CACHE TARGET_ARCH PROPERTY STRINGS x86 armv6)

find_package(Qt6 REQUIRED COMPONENTS Core Network HttpServer Bluetooth)

file(GLOB_RECURSE SRC_FILES
    "src/*.cpp"
    "include/*.hpp"
)

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        Qt6::Core
        Qt6::Network
        Qt6::HttpServer
        Qt6::Bluetooth
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    AUTOMOC ON
    AUTOUIC ON
    AUTORCC ON
    POSITION_INDEPENDENT_CODE OFF
    AUTOMOC_EXECUTABLE "/usr/lib/qt6/libexec/moc" 
)
