#include "diode_dataset.hpp"
#include "../utils.hpp"
#include "dataset.hpp"
#include <format>
#include <optional>
#include <regex>
#include <unordered_map>

std::size_t std::hash<DiodeDataPointID>::operator()(
	const DiodeDataPointID& dp
) const noexcept {
	return std::hash<bool>{}(dp.indoors) ^
		   std::hash<std::string>{}(dp.scene_id) ^
		   std::hash<std::string>{}(dp.scan_id) ^
		   std::hash<std::string>{}(dp.imgname);
}

std::filesystem::path DiodeDataPointID::get_evaluation_result_filename(
	const std::filesystem::path& evaluation_output_directory
) const {
	return evaluation_output_directory / (indoors ? "indoors" : "outdoor") /
		   std::format("{}_{}_{}_result.bin", scene_id, scan_id, imgname);
}

std::string DiodeDataPointID::to_string() const noexcept {
	return std::format(
		"{} scene {}, scan {}, image {}", indoors ? "indoors" : "outdoor",
		scene_id, scan_id, imgname
	);
}

tl::expected<RGBDImage, std::string> DiodeDataPoint::load(
	size_t depth_input_width,
	size_t depth_input_height
) const {
	auto image_result = load_rgb_image_file(
		image_filepath, depth_input_width, depth_input_height
	);
	if (!image_result)
		return tl::make_unexpected(image_result.error());

	auto depth_result = load_npy_file(depth_filepath);
	if (!depth_result)
		return tl::make_unexpected(depth_result.error());
	auto& depth = *depth_result;
	size_t expected_depth_size =
		DiodeDataset::DEPTH_WIDTH * DiodeDataset::DEPTH_HEIGHT;
	if (depth.size() != expected_depth_size) {
		return tl::unexpected(
			std::format(
				"Invalid depth size: expected {}, got {}", expected_depth_size,
				depth.size()
			)
		);
	}

	const auto depth_mask_result = load_npy_file(depth_mask_filepath);
	if (!depth_mask_result)
		return tl::make_unexpected(depth_mask_result.error());
	if (depth_mask_result->size() != expected_depth_size) {
		return tl::unexpected(
			std::format(
				"Invalid depth mask size: expected {}, got {}",
				expected_depth_size, depth_mask_result->size()
			)
		);
	}
	std::vector<bool> depth_mask(depth_mask_result->size());
	for (size_t i = 0; i < depth_mask.size(); ++i) {
		depth_mask[i] = (*depth_mask_result)[i] != 0.f;
	}

	return RGBDImage(
		depth_input_width, depth_input_height, std::move(*image_result),
		DiodeDataset::DEPTH_WIDTH, DiodeDataset::DEPTH_HEIGHT, std::move(depth),
		std::move(depth_mask)
	);
}

static std::optional<DiodeDataPointID>
match_image_file(const std::string& filename) {
	std::regex pattern(
		R"((\d+)_(\d+)_(outdoor|indoors)_(\w+)\.png)", std::regex::icase
	);
	std::smatch match;

	if (std::regex_match(filename, match, pattern)) {
		return DiodeDataPointID(
			match[3] == "indoors", match[1], match[2], match[4]
		);
	}
	return std::nullopt;
}

static std::optional<DiodeDataPointID>
match_depth_file(const std::string& filename) {
	std::regex pattern(
		R"((\d+)_(\d+)_(outdoor|indoors)_(\w+)_depth\.npy)", std::regex::icase
	);
	std::smatch match;

	if (std::regex_match(filename, match, pattern)) {
		return DiodeDataPointID(
			match[3] == "indoors", match[1], match[2], match[4]
		);
	}
	return std::nullopt;
}

static std::optional<DiodeDataPointID>
match_depth_mask_file(const std::string& filename) {
	std::regex pattern(
		R"((\d+)_(\d+)_(outdoor|indoors)_(\w+)_depth_mask\.npy)",
		std::regex::icase
	);
	std::smatch match;

	if (std::regex_match(filename, match, pattern)) {
		return DiodeDataPointID(
			match[3] == "indoors", match[1], match[2], match[4]
		);
	}
	return std::nullopt;
}

static std::optional<std::string>
match_scan_directory(const std::string& directory) {
	std::regex pattern(R"(scan_(\d+))", std::regex::icase);
	std::smatch match;

	if (std::regex_match(directory, match, pattern)) {
		return match[1];
	}
	return std::nullopt;
}

static std::unordered_map<std::string, std::filesystem::path>
search_for_scans_in_dataset(const std::filesystem::path& dataset_directory) {
	std::unordered_map<std::string, std::filesystem::path> scan_paths;

	for (const auto& entry :
		 std::filesystem::recursive_directory_iterator(dataset_directory)) {

		if (entry.is_directory()) {
			const auto& filepath = entry.path();
			const auto filename = filepath.filename();

			const std::optional<std::string> scan_id =
				match_scan_directory(filename);

			if (scan_id)
				scan_paths[*scan_id] = filepath;
		}
	}

	return scan_paths;
}

static std::vector<std::unique_ptr<RGBDDataPoint>>
search_for_datapoints_in_scan(const std::filesystem::path& scan_directory) {
	std::unordered_map<DiodeDataPointID, std::filesystem::path> image_filepaths;
	std::unordered_map<DiodeDataPointID, std::filesystem::path> depth_filepaths;
	std::unordered_map<DiodeDataPointID, std::filesystem::path>
		depth_mask_filepaths;

	for (const auto& entry :
		 std::filesystem::directory_iterator(scan_directory)) {

		const auto& filepath = entry.path();
		const auto filename = filepath.filename();
		if (!entry.is_regular_file())
			continue;

		const std::optional<DiodeDataPointID> image_data_point_id =
			match_image_file(filename);
		if (image_data_point_id) {
			image_filepaths[*image_data_point_id] = filepath;
			continue;
		}
		const std::optional<DiodeDataPointID> depth_data_point_id =
			match_depth_file(filename);
		if (depth_data_point_id) {
			depth_filepaths[*depth_data_point_id] = filepath;
			continue;
		}

		const std::optional<DiodeDataPointID> depth_mask_data_point_id =
			match_depth_mask_file(filename);
		if (depth_mask_data_point_id) {
			depth_mask_filepaths[*depth_mask_data_point_id] = filepath;
		} else if (filename.extension() == ".bin") {
			println_fmt("(Skipping file {})", filepath.string());
		}
	}

	std::vector<DiodeDataPoint> datapoints;
	for (const auto& [data_point_id, image_path] : image_filepaths) {
		if (depth_filepaths.contains(data_point_id)) {
			if (depth_mask_filepaths.contains(data_point_id)) {
				datapoints.emplace_back(
					data_point_id, image_path,
					depth_filepaths.at(data_point_id),
					depth_mask_filepaths.at(data_point_id)
				);
			} else {
				println_fmt(
					"(Skipping {} with no depth mask)",
					data_point_id.to_string()
				);
			}
		} else {
			println_fmt(
				"(Skipping {} with no depth)", data_point_id.to_string()
			);
		}
	}
	const auto datapoints_contains = [&](const DiodeDataPointID& id) {
		return std::ranges::any_of(
			datapoints,
			[&](const DiodeDataPoint& datapoint) { return datapoint.id == id; }
		);
	};
	for (const auto& [datapoint_id, depth_path] : depth_filepaths) {
		if (!datapoints_contains(datapoint_id)) {
			println_fmt(
				"(Skipping {} with no image or depth_mask)",
				datapoint_id.to_string()
			);
		}
	}
	for (const auto& [datapoint_id, depth_path] : depth_mask_filepaths) {
		if (!datapoints_contains(datapoint_id)) {
			println_fmt(
				"(Skipping {} with no image or depth)", datapoint_id.to_string()
			);
		}
	}

	std::vector<std::unique_ptr<RGBDDataPoint>> rgbd_datapoints(
		datapoints.size()
	);
	for (size_t i = 0; i < rgbd_datapoints.size(); i++) {
		rgbd_datapoints[i] = std::make_unique<DiodeDataPoint>(datapoints[i]);
	}

	return rgbd_datapoints;
}

std::vector<std::unique_ptr<RGBDDataPoint>>
DiodeDataset::scan(const std::filesystem::path& dataset_directory) const {
	std::unordered_map<std::string, std::filesystem::path> scan_paths =
		search_for_scans_in_dataset(dataset_directory);

	std::vector<std::unique_ptr<RGBDDataPoint>> datapoints;

	for (const auto& [scan_id, scan_directory] : scan_paths) {
		auto new_datapoints = search_for_datapoints_in_scan(scan_directory);
		datapoints.insert(
			datapoints.end(), std::make_move_iterator(new_datapoints.begin()),
			std::make_move_iterator(new_datapoints.end())
		);
		new_datapoints.clear();
	}

	return datapoints;
}