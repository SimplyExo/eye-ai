include(FetchContent)

# tensorflow-lite
if(NOT DEFINED TFLITE_RUNTIME_USE_PREBUILT_TFLITE)
	message(STATUS "TFLITE_RUNTIME_USE_PREBUILT_TFLITE is not defined, defaulting to OFF")
	set(TFLITE_RUNTIME_USE_PREBUILT_TFLITE OFF)
endif()

if(TFLITE_RUNTIME_USE_PREBUILT_TFLITE)
	message(STATUS "Using prebuilt TFLite library from LiteRT")
	set(TFLITE_INCLUDE_DIRS
		"${CMAKE_CURRENT_SOURCE_DIR}/litert-1.2.0/include"
		"${CMAKE_CURRENT_SOURCE_DIR}/litert-1.2.0/include/external/org_tensorflow"
		"${CMAKE_CURRENT_SOURCE_DIR}/litert-gpu-1.2.0/include"
		"${CMAKE_CURRENT_SOURCE_DIR}/qnn-litert-delegate-2.38.0/include"
	)
	set(TFLITE_LIBS
		"${CMAKE_CURRENT_SOURCE_DIR}/litert-1.2.0/lib/${TFLITE_RUNTIME_ABI}/libtensorflowlite_jni.so"
		"${CMAKE_CURRENT_SOURCE_DIR}/litert-gpu-1.2.0/lib/${TFLITE_RUNTIME_ABI}/libtensorflowlite_gpu_jni.so"
		"${CMAKE_CURRENT_SOURCE_DIR}/qnn-litert-delegate-2.38.0/lib/${TFLITE_RUNTIME_ABI}/libQnnTFLiteDelegate.so"
		"${CMAKE_CURRENT_SOURCE_DIR}/qnn-litert-delegate-2.38.0/lib/${TFLITE_RUNTIME_ABI}/libqnn_delegate_jni.so"
		"${CMAKE_CURRENT_SOURCE_DIR}/qnn-litert-delegate-2.38.0/lib/${TFLITE_RUNTIME_ABI}/libQnnHtp.so"
		"${CMAKE_CURRENT_SOURCE_DIR}/qnn-litert-delegate-2.38.0/lib/${TFLITE_RUNTIME_ABI}/libQnnHtpV79Stub.so"
		"${CMAKE_CURRENT_SOURCE_DIR}/qnn-litert-delegate-2.38.0/lib/${TFLITE_RUNTIME_ABI}/libQnnHtpV75Stub.so"
		"${CMAKE_CURRENT_SOURCE_DIR}/qnn-litert-delegate-2.38.0/lib/${TFLITE_RUNTIME_ABI}/libQnnHtpV73Stub.so"
		"${CMAKE_CURRENT_SOURCE_DIR}/qnn-litert-delegate-2.38.0/lib/${TFLITE_RUNTIME_ABI}/libQnnHtpV69Stub.so"
		"${CMAKE_CURRENT_SOURCE_DIR}/qnn-litert-delegate-2.38.0/lib/${TFLITE_RUNTIME_ABI}/libQnnHtpV68Stub.so"
		"${CMAKE_CURRENT_SOURCE_DIR}/qnn-litert-delegate-2.38.0/lib/${TFLITE_RUNTIME_ABI}/libQnnHtpPrepare.so"
		"${CMAKE_CURRENT_SOURCE_DIR}/qnn-litert-delegate-2.38.0/lib/${TFLITE_RUNTIME_ABI}/libQnnSystem.so"
	)
else()
	message(STATUS "Downloading tensorflow...")
	set(ABSL_PROPAGATE_CXX_STD ON)
	set(TFLITE_ENABLE_RUY ON)
	set(TFLITE_ENABLE_GPU ON)
	set(TFLITE_ENABLE_XNNPACK ON)
	set(TFLITE_C_BUILD_SHARED_LIBS OFF)
	FetchContent_Declare(
		tensorflow
		URL https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.19.0.zip
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
