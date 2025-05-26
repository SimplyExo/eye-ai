cmake_minimum_required(VERSION 3.10)

project(NativeLib)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# configure clang-tidy if installed
find_program(CLANG_TIDY_EXE NAMES clang-tidy)
if (CLANG_TIDY_EXE)
	set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif ()

add_subdirectory(third_party)

file(GLOB_RECURSE NATIVE_LIB_SOURCE_FILES
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
	"${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_library(
	NativeLib
	SHARED
	${NATIVE_LIB_SOURCE_FILES}
)

target_compile_features(NativeLib PUBLIC cxx_std_20)

target_include_directories(NativeLib PRIVATE
	"${CMAKE_CURRENT_SOURCE_DIR}/src"

	${THIRD_PARTY_INCLUDE_DIRS}

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
	${THIRD_PARTY_LIBS}
)
