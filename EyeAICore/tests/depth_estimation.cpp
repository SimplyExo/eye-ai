#include "npy.hpp"
#include "utils.hpp"
#include <tl/expected.hpp>

TEST(DepthEstimationTest, CorrectOutput) {
	constexpr size_t width = 256;
	constexpr size_t height = 256;
	constexpr auto test_image_path = "../tests/00022_00193_outdoor_010_030.png";
	constexpr auto expected_rel_depth_path =
		"../tests/00022_00193_outdoor_010_030_expected.npy";

	auto input_result = load_image_file(test_image_path, width, height);
	EXPECT_RESULT_HAS_VALUE(input_result);
	auto& input = *input_result;
	EXPECT_EQ(input.size(), 3 * width * height);

	linear_to_srgb(input);
	for (float& value : input) {
		value = std::clamp(value * 255.f, 0.f, 255.f);
	}

	const auto expected_output = npy::read_npy<float>(expected_rel_depth_path);

	auto depth_model_result = create_test_depth_model();
	EXPECT_RESULT_HAS_VALUE(depth_model_result);
	auto& depth_model = depth_model_result.value();

	std::vector<float> output(width * height);

	const auto run_error = depth_model->run(input, output);
	EXPECT_EQ(run_error, std::nullopt);

	// this might fail due to precision errors --> custom compare with tolerance
	EXPECT_EQ(output, expected_output.data);

	/* Uncomment, when the expected output changes:
	 * (npy file viewer: https://perchance.org/npy-file-viewer)

	npy::write_npy(
		"../tests/00022_00193_outdoor_010_030_expected.npy",
		npy::npy_data{.data = output, .shape = {height, width}}
	);

	*/
}
