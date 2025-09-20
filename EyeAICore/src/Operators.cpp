#include "EyeAICore/Operators.hpp"
#include "EyeAICore/MetricDepthModel.hpp"
#include "EyeAICore/utils/Profiling.hpp"

#include <algorithm>

FloatTensorBuffer<FloatTensorFormat::RelativeDepth>
raw_relative_depth_post_operator(
	FloatTensorBuffer<FloatTensorFormat::RawRelativeDepth>& input
) {
	PROFILE_DEPTH_FUNCTION()

	auto values = input.data();

	if (values.empty())
		return input.convert_format<FloatTensorFormat::RelativeDepth>();

	const auto [min_iter, max_iter] = std::ranges::minmax_element(values);
	const float min = *min_iter;
	const float max = *max_iter;

	const float diff = max - min;

	if (diff > 0.0f) {
		for (float& value : values) {
			value = (value - min) / diff;
		}
	} else {
		for (float& value : values) {
			value = 0.5f;
		}
	}

	return input.convert_format<FloatTensorFormat::RelativeDepth>();
}

FloatTensorBuffer<FloatTensorFormat::MetricDepth> rel2abs_operator(
	FloatTensorBuffer<FloatTensorFormat::RawRelativeDepth>& input,
	const std::array<float, MetricDepthModel::COEFFS_COUNT>& coeffs
) {
	PROFILE_DEPTH_FUNCTION()

	auto output = input.data();

	for (float& value : output) {
		const float relative_depth = value;
		const float absolute_depth =
			polynomial_n<4>(relative_depth, coeffs);
		value = absolute_depth;
	}

	return input.convert_format<FloatTensorFormat::MetricDepth>();
}

FloatTensorBuffer<FloatTensorFormat::MiDaSImageRGB>
midas_image_operator(FloatTensorBuffer<FloatTensorFormat::ImageRGB255>& input) {
	PROFILE_DEPTH_FUNCTION()

	constexpr static std::array<float, 3> MEAN = {123.675f, 116.28f, 103.53f};
	constexpr static std::array<float, 3> STDDEV = {58.395f, 57.12f, 57.375f};

	auto values = input.data();

	for (size_t i = 0; i + 2 < values.size(); i += 3) {
		values[i + 0] = (values[i + 0] - MEAN[0]) / STDDEV[0];
		values[i + 1] = (values[i + 1] - MEAN[1]) / STDDEV[1];
		values[i + 2] = (values[i + 2] - MEAN[2]) / STDDEV[2];
	}

	return input.convert_format<FloatTensorFormat::MiDaSImageRGB>();
}

FloatTensorBuffer<FloatTensorFormat::YoloImageRGB>
yolo_image_operator(FloatTensorBuffer<FloatTensorFormat::ImageRGB255>& input) {
	PROFILE_DEPTH_FUNCTION()

	auto values = input.data();

	for (float& value : values) {
		value /= 255.0f;
	}

	return input.convert_format<FloatTensorFormat::YoloImageRGB>();
}