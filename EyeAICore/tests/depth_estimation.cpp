#include "EyeAICore/TensorBuffer.hpp"
#include "npy.hpp"
#include "utils.hpp"
#include <tl/expected.hpp>

using ::testing::FloatNear;
using ::testing::Pointwise;

TEST(DepthEstimationTest, CorrectOutput) {
	constexpr float tolerance = 1e-3f;
	constexpr auto test_image_path = "../tests/00022_00193_outdoor_010_030.png";
	constexpr auto expected_rel_depth_path =
		"../tests/00022_00193_outdoor_010_030_expected.npy";

	auto depth_model_result = create_test_depth_model();
	EXPECT_RESULT_HAS_VALUE(depth_model_result);
	auto& metric_depth_model = depth_model_result.value();

	const auto input_shape = metric_depth_model->get_input_shape();
	assert(input_shape.size() == 4);
	assert(input_shape[0] == 1);
	assert(input_shape[3] == 3);
	int width = input_shape[2];
	int height = input_shape[1];

	auto input_result = load_image_file(test_image_path, width, height);
	EXPECT_RESULT_HAS_VALUE(input_result);
	auto& input = *input_result;
	EXPECT_EQ(input.data().size(), 3 * width * height);

	FloatTensorBuffer<FloatTensorFormat::ImageRGB255> input_tensor =
		image_rgb_255_operator(input);

	const auto expected_output = npy::read_npy<float>(expected_rel_depth_path);

	const auto run_result = metric_depth_model->run(input_tensor);
	EXPECT_RESULT_HAS_VALUE(run_result);
	const auto& output = run_result->data();

	// this might fail due to precision errors -> added tolerance
	EXPECT_THAT(output, Pointwise(FloatNear(tolerance), expected_output.data));

	/* Uncomment, when the expected output changes:
	 * (npy file viewer: https://perchance.org/npy-file-viewer)

	npy::write_npy(
		"../tests/00022_00193_outdoor_010_030_qualcomm.npy",//"../tests/00022_00193_outdoor_010_030_expected.npy",
		npy::npy_data_ptr{.data_ptr = output.data(), .shape = {(size_t)height, (size_t)width}}
	);

	*/
}
