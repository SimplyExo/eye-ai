cmake_minimum_required(VERSION 3.10)

project(EyeAICore)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# configure clang-tidy if installed
find_program(CLANG_TIDY_EXE NAMES clang-tidy)
if (CLANG_TIDY_EXE)
	set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif ()

if(DEFINED CMAKE_ANDROID_ARCH_ABI)
	set(EYE_AI_CORE_ABI ${CMAKE_ANDROID_ARCH_ABI})
else()
	message(STATUS "EyeAICore: Not using Android NDK, setting EyeAICore ABI to host arch")
	if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
		set(EYE_AI_CORE_ABI "x86_64")
	elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i386")
		set(EYE_AI_CORE_ABI "x86")
	elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
		set(EYE_AI_CORE_ABI "arm64-v8a")
	else()
		set(EYE_AI_CORE_ABI "armeabi-v7a")
	endif()
endif()

message(STATUS "EyeAICore ABI: ${EYE_AI_CORE_ABI}")


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
	${THIRD_PARTY_LIBS}
)
