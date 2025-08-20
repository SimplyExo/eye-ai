#include "sun_rgbd_dataset.hpp"
#include "../utils.hpp"
#include "EyeAICore/utils/Profiling.hpp"
#include <algorithm>
#include <filesystem>
#include <format>

std::filesystem::path SUN_RGBD_DataPoint::get_evaluation_result_filename(
	const std::filesystem::path& evaluation_output_directory
) const {
	return evaluation_output_directory / std::format("{}.bin", full_id);
}

tl::expected<RGBDImage, std::string> SUN_RGBD_DataPoint::load(
	size_t depth_input_width,
	size_t depth_input_height
) const {
	PROFILE_DEPTH_FUNCTION()

	auto image_result = load_rgb_image_file(
		image_filepath, depth_input_width, depth_input_height
	);
	if (!image_result)
		return tl::make_unexpected(image_result.error());
	auto image = image_rgb_255_operator(*image_result);

	size_t depth_width = 0;
	size_t depth_height = 0;
	auto depth_result = load_16bit_greyscale_image_file(
		depth_filepath, depth_width, depth_height
	);
	if (!depth_result)
		return tl::make_unexpected(depth_result.error());
	auto& depth = *depth_result;

	// SUN RGB-D encodes 13-bit depth values in 16-bit, unit: millimeters
	for (uint16_t& value : depth) {
		value = (value >> 3) | (value << (16 - 3));
	}

	// convert millimeters to meters
	std::vector<float> metric_depth(depth.size());
	for (size_t i = 0; i < depth.size(); ++i) {
		metric_depth[i] = std::min(static_cast<float>(depth[i]) / 1000.f, 8.f);
	}

	std::vector<bool> depth_mask(metric_depth.size());
	for (size_t i = 0; i < depth_mask.size(); ++i) {
		depth_mask[i] = depth[i] != 0;
	}

	return RGBDImage(
		depth_input_width, depth_input_height, std::move(image), depth_width,
		depth_height, std::move(metric_depth), std::move(depth_mask)
	);
}

static std::optional<std::filesystem::path>
get_first_file_in_directory(const std::filesystem::path& directory) {
	for (const auto& entry : std::filesystem::directory_iterator(directory)) {
		if (entry.is_regular_file()) {
			return entry.path();
		}
	}
	return std::nullopt;
}

/// iterates over directories in the given directory, adding datapoints if
/// "scene.txt" exists, and not recursively iterating over the found datapoint
/// directory further
static void iterator_directory_for_image_dir(
	const std::filesystem::path& directory,
	std::vector<std::unique_ptr<RGBDDataPoint>>& out_datapoints
) {
	PROFILE_DEPTH_FUNCTION()

	for (const auto& entry : std::filesystem::directory_iterator(directory)) {
		if (!entry.is_directory())
			continue;

		const auto scene_txt_filepath = entry.path() / "scene.txt";
		if (std::filesystem::exists(scene_txt_filepath)) {
			const auto relative_path =
				std::filesystem::relative(entry.path(), directory);

			const auto image_filepath =
				get_first_file_in_directory(entry.path() / "image");
			if (!image_filepath) {
				println_error_fmt(
					"no image file found in directory {}", entry.path().string()
				);
				return;
			}

			const auto depth_filepath =
				get_first_file_in_directory(entry.path() / "depth");
			if (!depth_filepath) {
				println_error_fmt(
					"no depth file found in directory {}", entry.path().string()
				);
				return;
			}

			std::string id = relative_path.string();
			std::ranges::replace(id, '/', '_');
			std::ranges::replace(id, '\\', '_');
			std::ranges::replace(id, '.', '_');
			out_datapoints.emplace_back(
				std::make_unique<SUN_RGBD_DataPoint>(
					id, *image_filepath, *depth_filepath
				)
			);
		} else {
			iterator_directory_for_image_dir(entry.path(), out_datapoints);
		}
	}
}

std::vector<std::unique_ptr<RGBDDataPoint>>
SUN_RGBD_Dataset::scan(const std::filesystem::path& dataset_directory) const {
	PROFILE_DEPTH_FUNCTION()

	std::vector<std::unique_ptr<RGBDDataPoint>> datapoints;

	iterator_directory_for_image_dir(dataset_directory, datapoints);

	return datapoints;
}