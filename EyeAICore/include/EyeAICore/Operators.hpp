#pragma once

#include "EyeAICore/TensorBuffer.hpp"

[[nodiscard]] FloatTensorBuffer<FloatTensorFormat::RelativeDepth>
raw_relative_depth_post_operator(
	FloatTensorBuffer<FloatTensorFormat::RawRelativeDepth>& input
);

[[nodiscard]] FloatTensorBuffer<FloatTensorFormat::MetricDepth>
rel2abs_operator(
	FloatTensorBuffer<FloatTensorFormat::RawRelativeDepth>& input,
	FloatTensorBuffer<FloatTensorFormat::Rel2AbsDepthCoefficientOutput>& coeffs
);

/// normalizes rgb input values (3 floats for r, g and b) based on their
/// mean and standard deviation values
[[nodiscard]] FloatTensorBuffer<FloatTensorFormat::MiDaSImageRGB>
midas_image_operator(FloatTensorBuffer<FloatTensorFormat::ImageRGB255>& input);

[[nodiscard]] FloatTensorBuffer<FloatTensorFormat::YoloImageRGB>
yolo_image_operator(FloatTensorBuffer<FloatTensorFormat::ImageRGB255>& input);