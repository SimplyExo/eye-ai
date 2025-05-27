#pragma once

#include "onnx/OnnxRuntime.hpp"
#include "tflite/TfLiteRuntime.hpp"
#include <span>

constexpr size_t RGB_CHANNELS = 3;

void run_depth_estimation(
	TfLiteRuntime& tflite_runtime,
	std::span<float> input,
	std::span<float> output,
	std::array<float, RGB_CHANNELS> mean,
	std::array<float, RGB_CHANNELS> stddev
);

void run_depth_estimation(
	OnnxRuntime& onnx_runtime,
	std::span<float> input_data,
	std::span<float> output_data,
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
void depth_colormap(
	std::span<const float> depth_values,
	std::span<int> colormapped_pixels
);