include(FetchContent)

# LiteRT (TFLite)
set(LITERT_INCLUDE_DIRS
	"${CMAKE_CURRENT_SOURCE_DIR}/litert-1.2.0/include"
	"${CMAKE_CURRENT_SOURCE_DIR}/litert-1.2.0/include/external/org_tensorflow"
)
set(LITERT_GPU_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/litert-gpu-1.2.0/include")
set(LITERT_LIB "${CMAKE_CURRENT_SOURCE_DIR}/litert-1.2.0/lib/${EYE_AI_CORE_ABI}/libtensorflowlite_jni.so")
set(LITERT_GPU_LIB "${CMAKE_CURRENT_SOURCE_DIR}/litert-gpu-1.2.0/lib/${EYE_AI_CORE_ABI}/libtensorflowlite_gpu_jni.so")

# OnnxRuntime
set(ONNXRUNTIME_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-android-1.22.0/include")
set(ONNXRUNTIME_LIB "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-android-1.22.0/lib/${EYE_AI_CORE_ABI}/libonnxruntime.so")

# tl::expected
set(EXPECTED_BUILD_TESTS OFF CACHE INTERNAL "")
set(EXPECTED_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
FetchContent_Declare(
	tl-expected
	GIT_REPOSITORY https://github.com/TartanLlama/expected.git
	GIT_TAG v1.1.0
)
FetchContent_MakeAvailable(tl-expected)

set(THIRD_PARTY_INCLUDE_DIRS
	${LITERT_INCLUDE_DIRS}
	${LITERT_GPU_INCLUDE_DIR}
	${ONNXRUNTIME_INCLUDE_DIR}
	PARENT_SCOPE
)
set(THIRD_PARTY_LIBS
	${LITERT_LIB}
	${LITERT_GPU_LIB}
	${ONNXRUNTIME_LIB}
	tl::expected
	PARENT_SCOPE
)