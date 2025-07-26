cmake_minimum_required(VERSION 3.22)

project(EyeAICore)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(EYE_AI_CORE_USE_PREBUILT_TFLITE "Use prebuilt TFLite library from LiteRT in third_party/litert(-gpu)-1.2.0 (requires EYE_AI_CORE_ABI)" OFF)

option(EYE_AI_CORE_ENABLE_ASAN "Enable AddressSanitizer" OFF)

option(EYE_AI_CORE_ENABLE_CLANG_TIDY "Enable Clang Tidy" ON)

option(EYE_AI_CORE_GENERATE_DOCS "Generate docs using doxygen" OFF)

option(EYE_AI_CORE_ENABLE_TESTS "Enable tests" OFF)

option(EYE_AI_CORE_ENABLE_TRACY_PROFILER "Enable tracy profiler" OFF)

if (EYE_AI_CORE_ENABLE_CLANG_TIDY)
	find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
	if (CLANG_TIDY_EXE STREQUAL "CLANG_TIDY_EXE-NOTFOUND")
		message(WARNING "clang-tidy not found, not using clang-tidy for static analysis despite EYE_AI_CORE_ENABLE_CLANG_TIDY begin ON")
		unset(CLANG_TIDY_EXE)
	endif ()
endif()

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
	set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif()

if (DEFINED CMAKE_ANDROID_ARCH_ABI)
    set(EYE_AI_CORE_USE_PREBUILT_TFLITE ON CACHE BOOL "Use prebuilt TFLite library from LiteRT in third_party/litert(-gpu)-1.2.0 (requires EYE_AI_CORE_ABI)" FORCE)
    set(EYE_AI_CORE_ABI ${CMAKE_ANDROID_ARCH_ABI})
else ()
    message(STATUS "EyeAICore: Not using Android NDK, setting EyeAICore ABI to host arch")
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        set(EYE_AI_CORE_ABI "x86_64")
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "i386")
        set(EYE_AI_CORE_ABI "x86")
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(EYE_AI_CORE_ABI "arm64-v8a")
    else ()
        set(EYE_AI_CORE_ABI "armeabi-v7a")
    endif ()
endif ()

message(STATUS "EyeAICore ABI: ${EYE_AI_CORE_ABI}")

if (EYE_AI_CORE_ENABLE_ASAN)
	message(STATUS "Enabling AddressSanitizer")
	add_compile_options(-fsanitize=address,undefined,leak -fno-omit-frame-pointer -fno-optimize-sibling-calls)
	add_link_options(-fsanitize=address,undefined,leak -static-libasan)
endif()

add_subdirectory(third_party)

file(GLOB_RECURSE EYE_AI_CORE_SOURCE_FILES
 "${CMAKE_CURRENT_SOURCE_DIR}/include/EyeAICore/*.hpp"
 "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_library(
 EyeAICore
 STATIC
 ${EYE_AI_CORE_SOURCE_FILES}
)

target_compile_features(EyeAICore PUBLIC cxx_std_20)
target_compile_options(EyeAICore PRIVATE
					$<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
										-Wall -Wextra -Werror -Wno-c99-extensions>
					$<$<CXX_COMPILER_ID:MSVC>:
										/W4 /WX>)

set_property(TARGET EyeAICore PROPERTY COMPILE_WARNING_AS_ERROR ON)

target_include_directories(EyeAICore
 PUBLIC
 "${CMAKE_CURRENT_SOURCE_DIR}/include"
 ${THIRD_PARTY_INCLUDE_DIRS}
)

target_link_libraries(
 EyeAICore
 PUBLIC
 ${THIRD_PARTY_LIBS}
)

if(EYE_AI_CORE_USE_PREBUILT_TFLITE)
    target_compile_definitions(EyeAICore PUBLIC EYE_AI_CORE_USE_PREBUILT_TFLITE=1)
else()
    target_compile_definitions(EyeAICore PUBLIC EYE_AI_CORE_USE_PREBUILT_TFLITE=0)
endif()

if (EYE_AI_CORE_ENABLE_CLANG_TIDY)
	if (DEFINED CLANG_TIDY_EXE)
		set_target_properties(EyeAICore PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
	endif ()
endif()

if (EYE_AI_CORE_ENABLE_TRACY_PROFILER)
	target_link_libraries(EyeAICore PUBLIC Tracy::TracyClient)
	target_compile_definitions(EyeAICore PUBLIC EYE_AI_CORE_ENABLE_TRACY_PROFILER=1)
else()
	target_compile_definitions(EyeAICore PUBLIC EYE_AI_CORE_ENABLE_TRACY_PROFILER=0)
endif()


if (EYE_AI_CORE_GENERATE_DOCS)
	find_package(Doxygen REQUIRED)

	include(FetchContent)
	message(STATUS "Downloading doxygen-awesome-css...")
	FetchContent_Declare(
		doxygen-awesome-css
		URL "https://github.com/jothepro/doxygen-awesome-css/archive/refs/tags/v2.3.4.zip"
	)
	FetchContent_MakeAvailable(doxygen-awesome-css)
	FetchContent_GetProperties(doxygen-awesome-css SOURCE_DIR AWESOME_CSS_DIR)

	# Doxyfile configuration
	set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/docs")
	set(DOXYGEN_FILE_PATTERNS *.cpp *.hpp *.md)
	set(DOXYGEN_RECURSIVE YES)
	set(DOXYGEN_EXTRACT_ALL YES)
	set(DOXYGEN_QUIET YES)
	set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
	set(DOXYGEN_ENABLE_PREPROCESSING YES)
	set(DOXYGEN_MACRO_EXPANSION YES)
	set(DOXYGEN_GENERATE_TREEVIEW YES)
	set(DOXYGEN_DISABLE_INDEX NO)
	set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
	set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
	set(DOXYGEN_LAYOUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/docs/DoxygenLayout.xml")
	set(DOXYGEN_HTML_EXTRA_STYLESHEET
		"${AWESOME_CSS_DIR}/doxygen-awesome.css"
		"${AWESOME_CSS_DIR}/doxygen-awesome-sidebar-only.css"
		"${AWESOME_CSS_DIR}/doxygen-awesome-sidebar-only-darkmode-toggle.css"
		"${CMAKE_CURRENT_SOURCE_DIR}/docs/custom.css"
	)
	set(DOXYGEN_HTML_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/docs/header.html")
	set(DOXYGEN_HTML_EXTRA_FILES
		"${AWESOME_CSS_DIR}/doxygen-awesome-darkmode-toggle.js"
		"${AWESOME_CSS_DIR}/doxygen-awesome-fragment-copy-button.js"
	)
	set(DOXYGEN_HTML_COPY_CLIPBOARD NO)
	set(DOXYGEN_HAVE_DOT YES)
	set(DOXYGEN_DOT_IMAGE_FORMAT svg)
	set(DOXYGEN_DOT_TRANSPARENT YES)

	doxygen_add_docs(
		EyeAICoreDoxygenDocs
		"${CMAKE_CURRENT_SOURCE_DIR}/README.md"
		"${CMAKE_CURRENT_SOURCE_DIR}/docs"
		"${CMAKE_CURRENT_SOURCE_DIR}/include/EyeAICore"
		"${CMAKE_CURRENT_SOURCE_DIR}/src"
		"${CMAKE_CURRENT_SOURCE_DIR}/metric_depth"
		COMMENT "Generating EyeAICore documentation with Doxygen"
	)
endif()

if (EYE_AI_CORE_ENABLE_TESTS)
	enable_testing()
	include(GoogleTest)

	add_executable(EyeAICoreTests
		"tests/depth_estimation.cpp"
		"tests/tflite_runtime.cpp"
		"tests/object_detection.cpp"
		"tests/stb_image.cpp"
	)
	target_include_directories(EyeAICoreTests PRIVATE ${LIBNPY_INCLUDE_DIR} ${STB_INCLUDE_DIR})
	target_link_libraries(EyeAICoreTests EyeAICore GTest::gtest_main GTest::gmock)
	gtest_discover_tests(EyeAICoreTests)
endif()

add_subdirectory(metric_depth)