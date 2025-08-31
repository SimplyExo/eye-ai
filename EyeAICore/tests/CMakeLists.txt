# places EyeAICoreTests executable into ./build/EyeAICoreTests where ctest can find it
set(CMAKE_CURRENT_BINARY_DIR ${CMAKE_BINARY_DIR})

enable_testing()
include(GoogleTest)

add_executable(EyeAICoreTests
	"depth_estimation.cpp"
	"object_detection.cpp"
	"metric_depth_estimation.cpp"
	"stb_image.cpp"
	"utils.hpp"
	"utils.cpp"
)
target_include_directories(EyeAICoreTests PRIVATE ${STB_INCLUDE_DIR} ${LIBNPY_INCLUDE_DIR})
target_link_libraries(EyeAICoreTests EyeAICore GTest::gtest_main GTest::gmock)
gtest_discover_tests(EyeAICoreTests)