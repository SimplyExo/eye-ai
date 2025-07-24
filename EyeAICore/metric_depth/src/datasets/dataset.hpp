#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <tl/expected.hpp>
#include <vector>

struct RGBDImage {
	/// one float for each channel (3: rgb) per pixel
	std::vector<float> rgb;

	/// one float per pixel
	std::vector<float> metric_depth;

	/// optional, one bool per pixel, true is valid, false is invalid
	std::optional<std::vector<bool>> depth_mask;
};

/// used to store all relevant info about a single RGBD image, able to load it
/// from disk
struct RGBDDataPoint {
	virtual ~RGBDDataPoint() = default;

	bool operator==(const RGBDDataPoint& other) const = default;

	[[nodiscard]] virtual std::filesystem::path get_evaluation_result_filename(
		const std::filesystem::path& evaluation_output_directory
	) const = 0;

	[[nodiscard]] virtual tl::expected<RGBDImage, std::string>
	load(size_t depth_input_width, size_t depth_input_height) const = 0;
};

class RGBDDataset {
  public:
	virtual ~RGBDDataset() = default;

	[[nodiscard]] virtual std::vector<std::unique_ptr<RGBDDataPoint>>
	scan(const std::filesystem::path& dataset_directory) = 0;

	[[nodiscard]] virtual size_t expected_image_count() const = 0;
};