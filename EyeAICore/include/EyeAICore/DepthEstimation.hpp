#pragma once

#include "tflite/TfLiteRuntime.hpp"
#include <span>
#include <tl/expected.hpp>

constexpr size_t RGB_CHANNELS = 3;

[[nodiscard]] tl::expected<void, std::string> run_depth_estimation(
	TfLiteRuntime& tflite_runtime,
	std::span<float> input,
	std::span<float> output,
	std::array<float, RGB_CHANNELS> mean,
	std::array<float, RGB_CHANNELS> stddev
);

/// normalizes rgb input values (3 floats for r, g and b) based on their mean
/// and standard deviation values
void normalize_rgb(
	std::span<float> values,
	std::array<float, RGB_CHANNELS> mean,
	std::array<float, RGB_CHANNELS> stddev
);

/// rescales values from [min, max] to [0, 1]
void min_max_scaling(std::span<float> values);

/// computes the int representation of the inferno colormap of the depth at each
/// pixel
[[nodiscard]] tl::expected<void, std::string> depth_colormap(
	std::span<const float> depth_values,
	std::span<int> colormapped_pixels
);