#pragma once

#include "dataset.hpp"

struct DiodeDataPointID {
	bool indoors = true;
	std::string scene_id;
	std::string scan_id;
	std::string imgname;

	[[nodiscard]] std::filesystem::path get_evaluation_result_filename(
		const std::filesystem::path& evaluation_output_directory
	) const;

	[[nodiscard]] std::string to_string() const noexcept;

	bool operator==(const DiodeDataPointID& other) const noexcept = default;
};

namespace std {
template<>
struct hash<DiodeDataPointID> {
	std::size_t operator()(const DiodeDataPointID& dp) const noexcept;
};
} // namespace std

struct DiodeDataPoint : public RGBDDataPoint {
	DiodeDataPointID id;

	std::filesystem::path image_filepath;
	std::filesystem::path depth_filepath;
	std::filesystem::path depth_mask_filepath;

	explicit DiodeDataPoint(
		DiodeDataPointID id,
		std::filesystem::path image_filepath,
		std::filesystem::path depth_filepath,
		std::filesystem::path depth_mask_filepath
	)
		: id(std::move(id)), image_filepath(std::move(image_filepath)),
		  depth_filepath(std::move(depth_filepath)),
		  depth_mask_filepath(std::move(depth_mask_filepath)) {}

	[[nodiscard]] std::filesystem::path get_evaluation_result_filename(
		const std::filesystem::path& evaluation_output_directory
	) const override {
		return id.get_evaluation_result_filename(evaluation_output_directory);
	}

	[[nodiscard]] tl::expected<RGBDImage, std::string>
	load(size_t depth_input_width, size_t depth_input_height) const override;
};

class DiodeDataset : public RGBDDataset {
  public:
	constexpr static size_t DEPTH_WIDTH = 1024;
	constexpr static size_t DEPTH_HEIGHT = 768;

	[[nodiscard]] std::vector<std::unique_ptr<RGBDDataPoint>>
	scan(const std::filesystem::path& dataset_directory) const override;

	[[nodiscard]] size_t expected_image_count() const override { return 771; }
};