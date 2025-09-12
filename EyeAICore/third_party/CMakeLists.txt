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
		"${CMAKE_CURRENT_SOURCE_DIR}/qnn-litert-delegate-2.38.0/include"
	)
	set(TFLITE_LIBS
		"${CMAKE_CURRENT_SOURCE_DIR}/litert-1.2.0/lib/${EYE_AI_CORE_ABI}/libtensorflowlite_jni.so"
		"${CMAKE_CURRENT_SOURCE_DIR}/litert-gpu-1.2.0/lib/${EYE_AI_CORE_ABI}/libtensorflowlite_gpu_jni.so"
		"${CMAKE_CURRENT_SOURCE_DIR}/qnn-litert-delegate-2.38.0/lib/${EYE_AI_CORE_ABI}/libQnnTFLiteDelegate.so"
		"${CMAKE_CURRENT_SOURCE_DIR}/qnn-litert-delegate-2.38.0/lib/${EYE_AI_CORE_ABI}/libqnn_delegate_jni.so"
		"${CMAKE_CURRENT_SOURCE_DIR}/qnn-litert-delegate-2.38.0/lib/${EYE_AI_CORE_ABI}/libQnnHtp.so"
		"${CMAKE_CURRENT_SOURCE_DIR}/qnn-litert-delegate-2.38.0/lib/${EYE_AI_CORE_ABI}/libQnnHtpV79Stub.so"
		"${CMAKE_CURRENT_SOURCE_DIR}/qnn-litert-delegate-2.38.0/lib/${EYE_AI_CORE_ABI}/libQnnHtpPrepare.so"
		"${CMAKE_CURRENT_SOURCE_DIR}/qnn-litert-delegate-2.38.0/lib/${EYE_AI_CORE_ABI}/libQnnSystem.so"
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

# libnpy
message(STATUS "Downloading libnpy...")
FetchContent_Declare(
	libnpy
	GIT_REPOSITORY https://github.com/llohse/libnpy.git
	GIT_TAG master
)
FetchContent_MakeAvailable(libnpy)
set(LIBNPY_INCLUDE_DIR "${libnpy_SOURCE_DIR}/include")

# stb (stb_image, stb_image_resize2)
message(STATUS "Downloading stb (stb_image, stb_image_resize2)...")
FetchContent_Declare(
	stb
	GIT_REPOSITORY https://github.com/nothings/stb.git
	GIT_TAG master
)
FetchContent_MakeAvailable(stb)
set(STB_INCLUDE_DIR ${stb_SOURCE_DIR})

# concurrentqueue
message(STATUS "Downloading concurrentqueue...")
FetchContent_Declare(
	concurrentqueue
	GIT_REPOSITORY    https://github.com/cameron314/concurrentqueue.git
	GIT_TAG           v1.0.4
)
FetchContent_MakeAvailable(concurrentqueue)

# eigen
message(STATUS "Downloading eigen...")
FetchContent_Declare(
	eigen
	GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
	GIT_TAG 3.4.0
	GIT_SHALLOW ON
)
FetchContent_MakeAvailable(eigen)

if (EYE_AI_CORE_ENABLE_TESTS)
	# libnpy
	message(STATUS "Downloading libnpy...")
	FetchContent_Declare(
		libnpy
		GIT_REPOSITORY https://github.com/llohse/libnpy.git
		GIT_TAG master
	)
	FetchContent_MakeAvailable(libnpy)
	set(LIBNPY_INCLUDE_DIR "${libnpy_SOURCE_DIR}/include" PARENT_SCOPE)

	# stb (stb_image, stb_image_resize2)
	message(STATUS "Downloading stb (stb_image, stb_image_resize2)...")
	FetchContent_Declare(
		stb
		GIT_REPOSITORY https://github.com/nothings/stb.git
		GIT_TAG master
	)
	FetchContent_MakeAvailable(stb)
	set(STB_INCLUDE_DIR ${stb_SOURCE_DIR} PARENT_SCOPE)

	# GTest
	message(STATUS "Downloading googletest...")
	FetchContent_Declare(
	  googletest
	  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
	)
	FetchContent_MakeAvailable(googletest)
endif()

if (EYE_AI_CORE_ENABLE_TRACY_PROFILER)
	# tracy
	message(STATUS "Downloading tracy...")
	option(TRACY_ENABLE "" ON)
	FetchContent_Declare(
		tracy
		URL https://github.com/wolfpld/tracy/archive/refs/tags/v0.12.2.zip
	)
	FetchContent_MakeAvailable(tracy)
endif ()


FetchContent_Declare(
  openal-soft
  GIT_REPOSITORY https://github.com/kcat/openal-soft.git
  GIT_TAG master
)

FetchContent_MakeAvailable(openal-soft)

FetchContent_Declare(
  libsndfile
  GIT_REPOSITORY https://github.com/libsndfile/libsndfile.git
  GIT_TAG master
)

FetchContent_MakeAvailable(libsndfile)

if (NOT TARGET nlohmann_json)
    message(STATUS "Fetching nlohmann_json via FetchContent...")
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2
    )
    FetchContent_MakeAvailable(nlohmann_json)
else()
    message(STATUS "nlohmann_json already provided (Prefab/Gradle)")
endif()

if(EYE_AI_CORE_FORCE_OBOE)
	find_package(oboe REQUIRED CONFIG)
	target_compile_definitions(OpenAL PRIVATE "ALSOFT_DRIVERS=oboe")
endif()

set(THIRD_PARTY_INCLUDE_DIRS
	${TFLITE_INCLUDE_DIRS}
	${LIBNPY_INCLUDE_DIR}
	${STB_INCLUDE_DIR}
	PARENT_SCOPE
)
set(THIRD_PARTY_LIBS
	${TFLITE_LIBS}
	tl::expected
	concurrentqueue
	OpenAL
	SndFile::sndfile
	nlohmann_json::nlohmann_json
	PARENT_SCOPE
)

