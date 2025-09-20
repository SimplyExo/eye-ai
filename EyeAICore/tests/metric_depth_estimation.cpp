#include "EyeAICore/MetricDepthModel.hpp"
#include "utils.hpp"
#include <cstddef>
#include <gtest/gtest.h>

TEST(MetricDepthEstimationTest, TestMetricDepthEstimation) {
	constexpr float tolerance = 1e-3f;
	constexpr auto test_image_path = "../tests/0000103.jpg";
	constexpr auto expected_abs_depth_path = "../tests/0000103.png";

	auto metric_depth_model_result = create_test_metric_depth_model();
	EXPECT_RESULT_HAS_VALUE(metric_depth_model_result);
	auto& metric_depth_model = metric_depth_model_result.value();

	const size_t width = 256;
	const size_t height = 256;

	auto input_result = load_image_file(test_image_path, width, height);
	EXPECT_RESULT_HAS_VALUE(input_result);
	auto& input = *input_result;
	EXPECT_EQ(input.data().size(), 3 * width * height);

	auto input_tensor = image_rgb_255_operator(input);

	const auto pixel_count = width * height;

	std::vector<float> output(pixel_count);

	const auto run_result = metric_depth_model->run(input_tensor);
	EXPECT_RESULT_HAS_VALUE(run_result);
}

/// test if horner's method of a 4 degree polynomial function is
/// implemented correctly
TEST(MetricDepthEstimationTest, TestPolynomial) {
	const static size_t COUNT = 10'000;
	const std::array<float, MetricDepthModel::COEFFS_COUNT> coeffs{
		1.f, 2.f, 3.f, 4.f, 5.f
	};

	// f(x) = a4 * x⁴ + a3 * x³ + a2 * x² + a1 * x + a0
	auto naive_polynomial_4 = [](float x, const std::array<float, 5>& coeffs) {
		const float xx = x * x;
		const float xxx = xx * x;
		const float xxxx = xxx * x;
		return (coeffs[4] * xxxx) + (coeffs[3] * xxx) + (coeffs[2] * xx) +
			   (coeffs[1] * x) + coeffs[0];
	};

	for (size_t i = 0; i < COUNT; ++i) {
		const float x = (float)i - ((float)COUNT / 2.f);
		const float naive = naive_polynomial_4(x, coeffs);
		const float horner = polynomial_4(x, coeffs);
		const float horner_general = polynomial_n<4>(x, coeffs);
		EXPECT_FLOAT_EQ(naive, horner);
		EXPECT_FLOAT_EQ(naive, horner_general);
	}
}