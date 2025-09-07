project(audio)

add_executable(AudioTest "main.cpp")

target_compile_features(AudioTest PUBLIC cxx_std_20)

target_link_libraries(AudioTest PRIVATE EyeAICore)


target_link_libraries(AudioTest PRIVATE OpenAL)

target_link_libraries(AudioTest PRIVATE SndFile::sndfile)

target_link_libraries(AudioTest PRIVATE nlohmann_json::nlohmann_json)
