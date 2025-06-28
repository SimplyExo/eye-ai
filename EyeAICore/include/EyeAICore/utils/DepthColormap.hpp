#pragma once

#include <optional>
#include <span>
#include <string>

struct [[nodiscard]] DepthColorArraySizeMismatch {
	size_t depth_values_size;
	size_t colormapped_pixels_size;

	[[nodiscard]] std::string to_string() const;
};

/// computes the int representation of the inferno colormap of the depth at each
/// pixel
[[nodiscard]] std::optional<DepthColorArraySizeMismatch> depth_colormap(
	std::span<const float> depth_values,
	std::span<int> colormapped_pixels
);