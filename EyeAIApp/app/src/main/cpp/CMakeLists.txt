cmake_minimum_required(VERSION 3.10)

project(NativeLib)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# configure clang-tidy if installed
find_program(CLANG_TIDY_EXE NAMES clang-tidy)
if (CLANG_TIDY_EXE)
	set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif ()

include(FetchContent)

FetchContent_Declare(
	nlohmann_json
	GIT_REPOSITORY https://github.com/nlohmann/json.git
	GIT_TAG v3.12.0
)

FetchContent_MakeAvailable(nlohmann_json)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../../../EyeAICore" "${CMAKE_BINARY_DIR}/EyeAICore")

add_library(
	NativeLib
	SHARED
	"${CMAKE_CURRENT_SOURCE_DIR}/NativeLib.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/ImageUtils.cpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/NativeJavaScopes.cpp"
)

target_compile_features(NativeLib PUBLIC cxx_std_20)

target_include_directories(NativeLib PRIVATE
	"${CMAKE_SYSROOT}/usr/include/"
	"${CMAKE_SYSROOT}/usr/include/c++/v1"
	"${CMAKE_SYSROOT}/usr/include/${ANDROID_TOOLCHAIN_NAME}"
)

target_link_libraries(
	NativeLib
	android
	log
	jnigraphics
	EyeAICore
	nlohmann_json::nlohmann_json
)
