#pragma once

#include "dataset.hpp"

struct SUN_RGBD_DataPoint : public RGBDDataPoint {
	std::string full_id;

	std::filesystem::path image_filepath;
	std::filesystem::path depth_filepath;

	SUN_RGBD_DataPoint(
		std::string id,
		std::filesystem::path image_filepath,
		std::filesystem::path depth_filepath
	)
		: full_id(std::move(id)), image_filepath(std::move(image_filepath)),
		  depth_filepath(std::move(depth_filepath)) {}

	[[nodiscard]] tl::expected<RGBDImage, std::string>
	load(size_t depth_input_width, size_t depth_input_height) const override;
};

class SUN_RGBD_Dataset : public RGBDDataset {
  public:
	[[nodiscard]] std::vector<std::unique_ptr<RGBDDataPoint>>
	scan(const std::filesystem::path& dataset_directory) const override;

	[[nodiscard]] size_t expected_image_count() const override { return 10335; }
};