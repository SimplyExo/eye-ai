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

	size_t channel = 0;

	for (float& value : values) {
		value = (value - mean[channel]) / stddev[channel];
		channel = (channel + 1) % 3;
	}

	return std::nullopt;
}