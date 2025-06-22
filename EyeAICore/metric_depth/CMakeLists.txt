project(metric_depth)

add_executable(EvaluateDataset "src/main.cpp")

target_compile_features(EvaluateDataset PUBLIC cxx_std_20)

target_link_libraries(EvaluateDataset PRIVATE EyeAICore)
