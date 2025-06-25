project(object_detection)

add_executable(object_detection "src/test_yolo.cpp")

target_compile_features(object_detection PUBLIC cxx_std_20)

target_link_libraries(object_detection PRIVATE EyeAICore)
