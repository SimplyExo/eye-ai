#pragma once

#include "EyeAICore/TensorBuffer.hpp"
#include <filesystem>
#include <optional>
#include <string>
#include <tl/expected.hpp>
#include <vector>

struct RGBDImage {
	size_t rgb_width;
	size_t rgb_height;

	/// one float for each channel (3: rgb) per pixel
	FloatTensorBuffer<FloatTensorFormat::ImageRGB255> rgb;

	size_t depth_width;
	size_t depth_height;

	/// one float per pixel
	std::vector<float> metric_depth;

	/// optional, one bool per pixel, true is valid, false is invalid
	std::optional<std::vector<bool>> depth_mask;
};

/// used to store all relevant info about a single RGBD image, able to load it
/// from disk
struct RGBDDataPoint {
	virtual ~RGBDDataPoint() = default;

	[[nodiscard]] virtual tl::expected<RGBDImage, std::string>
	load(size_t depth_input_width, size_t depth_input_height) const = 0;

	bool operator==(const RGBDDataPoint& other) const = default;
};

class RGBDDataset {
  public:
	virtual ~RGBDDataset() = default;

	[[nodiscard]] virtual std::vector<std::unique_ptr<RGBDDataPoint>>
	scan(const std::filesystem::path& dataset_directory) const = 0;

	[[nodiscard]] virtual size_t expected_image_count() const = 0;
};