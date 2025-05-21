cmake_minimum_required(VERSION 3.10)

project(NativeLib)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# configure clang-tidy if installed
find_program(CLANG_TIDY_EXE NAMES clang-tidy)
if (CLANG_TIDY_EXE)
	set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif ()


set(LITERT_INCLUDE_DIRS
	"${CMAKE_CURRENT_SOURCE_DIR}/third_party/litert-1.2.0/include"
	"${CMAKE_CURRENT_SOURCE_DIR}/third_party/litert-1.2.0/include/external/org_tensorflow"
)
set(LITERT_GPU_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/litert-gpu-1.2.0/include")
set(LITERT_LIB "${CMAKE_CURRENT_SOURCE_DIR}/third_party/litert-1.2.0/lib/${ANDROID_ABI}/libtensorflowlite_jni.so")
set(LITERT_GPU_LIB "${CMAKE_CURRENT_SOURCE_DIR}/third_party/litert-gpu-1.2.0/lib/${ANDROID_ABI}/libtensorflowlite_gpu_jni.so")

set(ONNXRUNTIME_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-android-1.22.0/include")
set(ONNXRUNTIME_LIB "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-android-1.22.0/lib/${ANDROID_ABI}/libonnxruntime.so")

add_library(
	NativeLib
	SHARED
	"${CMAKE_CURRENT_SOURCE_DIR}/src/NativeLib.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/DepthEstimation.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/DepthEstimation.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/utils/Log.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/utils/Exceptions.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/utils/NativeJavaScopes.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/utils/Profiling.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/utils/Profiling.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/utils/ImageUtils.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/utils/ImageUtils.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/tflite/TfLiteUtils.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/tflite/TfLiteRuntime.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/tflite/TfLiteRuntime.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/onnx/OnnxUtils.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/onnx/OnnxRuntime.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/onnx/OnnxRuntime.cpp"
)

target_compile_features(NativeLib PUBLIC cxx_std_20)

target_include_directories(NativeLib PUBLIC
	"${CMAKE_CURRENT_SOURCE_DIR}/src"

	${LITERT_INCLUDE_DIRS}
	${LITERT_GPU_INCLUDE_DIR}

	${ONNXRUNTIME_INCLUDE_DIR}

								# just so that vscode knows where to look for headers
	"${CMAKE_SYSROOT}/usr/include/"
	"${CMAKE_SYSROOT}/usr/include/c++/v1"
	"${CMAKE_SYSROOT}/usr/include/${ANDROID_TOOLCHAIN_NAME}"
)

target_link_libraries(
	NativeLib
	android
	log
	jnigraphics
	${LITERT_LIB}
	${LITERT_GPU_LIB}
	${ONNXRUNTIME_LIB}
)
