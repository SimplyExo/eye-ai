cmake_minimum_required(VERSION 3.22)

project(EyeAICore)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(EYE_AI_CORE_USE_PREBUILT_TFLITE "Use prebuilt TFLite library from LiteRT in third_party/litert(-gpu)-1.2.0 (requires EYE_AI_CORE_ABI)" OFF)

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

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

if (ENABLE_ASAN)
	message(STATUS "Enabling AddressSanitizer")
	add_compile_options(-fsanitize=address)
	add_link_options(-fsanitize=address -static-libasan)
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

add_subdirectory(metric_depth)
