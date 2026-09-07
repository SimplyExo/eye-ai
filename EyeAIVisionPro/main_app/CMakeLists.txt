cmake_minimum_required(VERSION 3.23)

project(EyeAIVision_Main
    VERSION 1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(TARGET_ARCH "x86" CACHE STRING "Target architecture: x86 or armv6")

set_property(CACHE TARGET_ARCH PROPERTY STRINGS x86 armv6)

if(TARGET_ARCH STREQUAL "x86")

    message(STATUS "Building for x86")

elseif(TARGET_ARCH STREQUAL "armv6")

    message(STATUS "Building for ARMv6")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv6")

else()

    message(FATAL_ERROR
        "Unsupported TARGET_ARCH='${TARGET_ARCH}'. "
        "Use 'x86' or 'armv6'."
    )

endif()

find_package(Qt6 REQUIRED COMPONENTS Core Network)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
)

file(GLOB_RECURSE SRC_FILES
    "src/*.cpp"
    "include/*.hpp"
)

add_executable(${PROJECT_NAME} ${SRC_FILES})

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        Qt6::Core
        Qt6::Network
)

set_target_properties(EyeAIVision_Main PROPERTIES
    AUTOMOC ON
    AUTOUIC ON
    AUTORCC ON
    AUTOMOC_EXECUTABLE "/usr/lib/qt6/libexec/moc"
)
