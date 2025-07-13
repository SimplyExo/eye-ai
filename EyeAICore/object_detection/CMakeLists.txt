cmake_minimum_required(VERSION 3.16)

project(object_detection)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Quelle und Ziel definieren
add_executable(object_detection "src/test_yolo.cpp")

# C++ Features aktivieren
target_compile_features(object_detection PUBLIC cxx_std_20)

# Include-Verzeichnisse von OpenCV hinzufügen
target_include_directories(object_detection PRIVATE ${OpenCV_INCLUDE_DIRS})

# Bibliotheken verlinken
target_link_libraries(object_detection PRIVATE EyeAICore ${OpenCV_LIBS})
