project(audio)

add_executable(AudioTest "main.cpp")

target_compile_features(AudioTest PUBLIC cxx_std_20)

target_link_libraries(AudioTest PRIVATE EyeAICore)

include(FetchContent)

FetchContent_Declare(
  openal-soft
  GIT_REPOSITORY https://github.com/kcat/openal-soft.git
  GIT_TAG master 
)

FetchContent_MakeAvailable(openal-soft)

target_link_libraries(AudioTest PRIVATE OpenAL)
