#include "EyeAICore/Operators.hpp"
#include "EyeAICore/utils/Profiling.hpp"

#include <algorithm>

std::optional<OperatorError>
MinMaxOperator::execute(std::span<float> values) const {
	PROFILE_DEPTH_SCOPE("MinMaxOperator")

	if (values.empty())
		return std::nullopt;

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

	return std::nullopt;
}

std::optional<OperatorError>
RgbNormalizeOperator::execute(std::span<float> values) const {
	PROFILE_DEPTH_SCOPE("RgbNormalizeOperator")

	if (values.size() % 3 != 0)
		return OperatorError::fmt(
			"Invalid values size of {}, it is not a multiple of 3",
			values.size()
		);

	for (size_t i = 0; i < values.size(); i += 3) {
		values[i + 0] = (values[i + 0] - mean[0]) / stddev[0];
		values[i + 1] = (values[i + 1] - mean[1]) / stddev[1];
		values[i + 2] = (values[i + 2] - mean[2]) / stddev[2];
	}

	return std::nullopt;
}

std::optional<OperatorError>
RgbNormalizeOperatorYolo::execute(std::span<float> values) const {
	PROFILE_DEPTH_SCOPE("RgbNormalizeOperatorYolo")

	if (values.size() % 3 != 0)
		return OperatorError::fmt(
			"Invalid values size of {}, it is not a multiple of 3",
			values.size()
		);

	for (size_t i = 0; i < values.size(); i += 3) {
		values[i + 0] = values[i + 0] / 255.0f;
		values[i + 1] = values[i + 1] / 255.0f;
		values[i + 2] = values[i + 2] / 255.0f;
	}

	return std::nullopt;
}

