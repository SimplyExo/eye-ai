cmake_minimum_required(VERSION 3.10)

project(NativeLib)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# configure clang-tidy if installed
find_program(CLANG_TIDY_EXE NAMES clang-tidy)
if (CLANG_TIDY_EXE)
	set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif ()

add_library(
	NativeLib
	SHARED
	"${CMAKE_CURRENT_SOURCE_DIR}/NativeLib.cpp"
)

target_compile_features(NativeLib PUBLIC cxx_std_20)

target_link_libraries(
	NativeLib
	android
	log
	jnigraphics
)

# use 16 KB ELF aligment
# (see https://developer.android.com/guide/practices/page-sizes)
target_link_options(NativeLib PRIVATE
	"-Wl,-z,max-page-size=16384"
	"-Wl,-z,common-page-size=16384"
)
