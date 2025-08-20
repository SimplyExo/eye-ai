enable_testing()
include(GoogleTest)

add_executable(EyeAICoreTests
	"depth_estimation.cpp"
	"object_detection.cpp"
	"stb_image.cpp"
	"utils.hpp"
	"utils.cpp"
)
target_include_directories(EyeAICoreTests PRIVATE ${LIBNPY_INCLUDE_DIR})
target_link_libraries(EyeAICoreTests EyeAICore GTest::gtest_main GTest::gmock)
gtest_discover_tests(EyeAICoreTests)