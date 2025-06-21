include(FetchContent)

# tensorflow-lite
if(NOT DEFINED EYE_AI_CORE_USE_PREBUILT_TFLITE)
	message(STATUS "EYE_AI_CORE_USE_PREBUILT_TFLITE is not defined, defaulting to OFF")
	set(EYE_AI_CORE_USE_PREBUILT_TFLITE OFF)
endif()

if(EYE_AI_CORE_USE_PREBUILT_TFLITE)
	message(STATUS "Using prebuilt TFLite library from LiteRT")
	set(TFLITE_INCLUDE_DIRS
		"${CMAKE_CURRENT_SOURCE_DIR}/litert-1.2.0/include"
		"${CMAKE_CURRENT_SOURCE_DIR}/litert-1.2.0/include/external/org_tensorflow"
		"${CMAKE_CURRENT_SOURCE_DIR}/litert-gpu-1.2.0/include"
	)
	set(TFLITE_LIBS
		"${CMAKE_CURRENT_SOURCE_DIR}/litert-1.2.0/lib/${EYE_AI_CORE_ABI}/libtensorflowlite_jni.so"
		"${CMAKE_CURRENT_SOURCE_DIR}/litert-gpu-1.2.0/lib/${EYE_AI_CORE_ABI}/libtensorflowlite_gpu_jni.so"
	)
else()
	message(STATUS "Downloading tensorflow...")
	set(ABSL_PROPAGATE_CXX_STD ON)
	set(TFLITE_ENABLE_GPU ON)
	set(TFLITE_ENABLE_XNNPACK ON)
	set(TFLITE_C_BUILD_SHARED_LIBS OFF)
	FetchContent_Declare(
		tensorflow
		GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
		GIT_TAG v2.19.0
	)
	FetchContent_MakeAvailable(tensorflow)
	add_subdirectory(
		"${tensorflow_SOURCE_DIR}/tensorflow/lite/c"
		"${CMAKE_CURRENT_BINARY_DIR}/tensorflowlite_c"
		EXCLUDE_FROM_ALL
	)
	set(TFLITE_LIBS tensorflowlite_c absl::log)
endif()

# tl::expected
set(EXPECTED_BUILD_TESTS OFF CACHE INTERNAL "")
set(EXPECTED_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
message(STATUS "Downloading tf::expected...")
FetchContent_Declare(
	tl-expected
	GIT_REPOSITORY https://github.com/TartanLlama/expected.git
	GIT_TAG v1.1.0
)
FetchContent_MakeAvailable(tl-expected)

set(THIRD_PARTY_INCLUDE_DIRS
	${TFLITE_INCLUDE_DIRS}
	PARENT_SCOPE
)
set(THIRD_PARTY_LIBS
	${TFLITE_LIBS}
	tl::expected
	PARENT_SCOPE
)
