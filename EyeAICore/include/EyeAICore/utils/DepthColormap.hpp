#pragma once

#include <span>
#include <string>
#include <tl/expected.hpp>

/// computes the int representation of the inferno colormap of the depth at each
/// pixel
[[nodiscard]] tl::expected<void, std::string> depth_colormap(
	std::span<const float> depth_values,
	std::span<int> colormapped_pixels
);