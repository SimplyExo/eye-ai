#include "EyeAICore/Operators.hpp"
#include "EyeAICore/utils/Profiling.hpp"

#include <algorithm>

std::optional<OperatorError>
RelativeDepthPostOperator::execute(std::span<float> input) const {
	PROFILE_DEPTH_SCOPE("RelativeDepthPostOperator")

	if (input.empty())
		return std::nullopt;

	const auto [min_iter, max_iter] = std::ranges::minmax_element(input);
	const float min = *min_iter;
	const float max = *max_iter;

	const float diff = max - min;

	if (diff > 0.0f) {
		for (float& value : input) {
			value = (value - min) / diff;
		}
	} else {
		for (float& value : input) {
			value = 0.5f;
		}
	}

	return std::nullopt;
}

std::optional<OperatorError>
MiDaSImageOperator::execute(std::span<float> input) const {
	PROFILE_DEPTH_SCOPE("MiDaSImageOperator")

	if (input.size() % 3 != 0)
		return OperatorError(
			std::format(
				"Invalid values size of {}, it is not a multiple of 3",
				input.size()
			)
		);

	for (size_t i = 0; i < input.size(); i += 3) {
		input[i + 0] = (input[i + 0] - MEAN[0]) / STDDEV[0];
		input[i + 1] = (input[i + 1] - MEAN[1]) / STDDEV[1];
		input[i + 2] = (input[i + 2] - MEAN[2]) / STDDEV[2];
	}

	return std::nullopt;
}

std::optional<OperatorError>
YoloImageOperator::execute(std::span<float> input) const {
	PROFILE_DEPTH_SCOPE("YoloImageOperator")

	for (float& value : input) {
		value /= 255.0f;
	}

	return std::nullopt;
}

std::string_view format_float_tensor_format(FloatTensorFormat format) {
	switch (format) {
	case FloatTensorFormat::ImageRGB255Float:
		return "ImageRGB255Float";
	case FloatTensorFormat::MiDaSImageRGBFloat:
		return "MiDaSImageRGBFloat";
	case FloatTensorFormat::YoloImageRGBFloat:
		return "YoloImageRGBFloat";
	case FloatTensorFormat::RelativeDepth:
		return "RelativeDepth";
	case FloatTensorFormat::RawRelativeDepth:
		return "RawRelativeDepth";
	case FloatTensorFormat::YoloOutput:
		return "YoloOutput";
	default: 
		return "invalid";
	}

}