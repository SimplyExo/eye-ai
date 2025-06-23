#pragma once

#include "EyeAICore/DepthModel.hpp"
#include "EyeAICore/utils/Errors.hpp"
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <regex>
#include <span>
#include <tl/expected.hpp>

template<typename T>
static tl::expected<std::vector<T>, std::string>
read_binary_file(const std::filesystem::path& filepath) {
	std::ifstream file(filepath, std::ios::binary | std::ios::ate);

	if (!file.is_open())
		return tl::unexpected_fmt("Failed to open file: {}", filepath.string());

	std::streamsize binary_size = file.tellg();
	file.seekg(0, std::ios::beg);

	if (binary_size % sizeof(T) != 0) {
		return tl::unexpected_fmt(
			"File size {} is not a multiple of sizeof({})", binary_size,
			typeid(T).name()
		);
	}

	std::vector<T> buffer(binary_size / sizeof(T));

	if (!file.read(reinterpret_cast<char*>(buffer.data()), binary_size))
		return tl::unexpected_fmt("Failed to read file: {}", filepath.string());

	return buffer;
}

[[nodiscard]] static tl::expected<void, std::string>
save_evaluation_result_file(
	const std::filesystem::path& filepath,
	std::span<const float> relative_absolute_pairs
) {
	std::filesystem::create_directories(filepath.parent_path());
	std::ofstream file(filepath);
	if (!file.is_open())
		return tl::unexpected_fmt("Failed to open file: {}", filepath.string());

	for (size_t i = 0; i < relative_absolute_pairs.size(); i += 2) {
		file << std::format(
			"{},{}\n", relative_absolute_pairs[i],
			relative_absolute_pairs[i + 1]
		);
	}

	file.flush();

	return {};
}

template<typename... Args>
void println_fmt(const std::format_string<Args...> fmt, Args&&... args) {
	const std::string formatted = std::vformat(
		fmt.get(), std::make_format_args(std::forward<Args>(args)...)
	);

	std::cout << formatted << '\n';
}

template<typename... Args>
void println_error_fmt(const std::format_string<Args...> fmt, Args&&... args) {
	const std::string formatted = std::vformat(
		fmt.get(), std::make_format_args(std::forward<Args>(args)...)
	);

	std::cerr << formatted << '\n';
}

struct DataPoint {
	bool indoors = true;
	std::string scene_id;
	std::string scan_id;
	std::string imgname;

	bool operator==(const DataPoint& other) const = default;

	[[nodiscard]] std::string to_string() const noexcept {
		return std::format(
			"{} scene {}, scan {}, image {}", indoors ? "indoors" : "outdoor",
			scene_id, scan_id, imgname
		);
	}
};

namespace std {
template<>
struct hash<DataPoint> {
	std::size_t operator()(const DataPoint& dp) const noexcept {
		return std::hash<bool>{}(dp.indoors) ^
			   std::hash<std::string>{}(dp.scene_id) ^
			   std::hash<std::string>{}(dp.scan_id) ^
			   std::hash<std::string>{}(dp.imgname);
	}
};
} // namespace std

static std::optional<DataPoint> match_image_file(const std::string& filename) {
	std::regex pattern(
		R"((\d+)_(\d+)_(outdoor|indoors)_(\w+)_image\.bin)", std::regex::icase
	);
	std::smatch match;

	if (std::regex_match(filename, match, pattern)) {
		return DataPoint(match[3] == "indoors", match[1], match[2], match[4]);
	}
	return std::nullopt;
}

static std::optional<DataPoint> match_depth_file(const std::string& filename) {
	std::regex pattern(
		R"((\d+)_(\d+)_(outdoor|indoors)_(\w+)_depth\.bin)", std::regex::icase
	);
	std::smatch match;

	if (std::regex_match(filename, match, pattern)) {
		return DataPoint(match[3] == "indoors", match[1], match[2], match[4]);
	}
	return std::nullopt;
}

static std::optional<DataPoint>
match_depth_mask_file(const std::string& filename) {
	std::regex pattern(
		R"((\d+)_(\d+)_(outdoor|indoors)_(\w+)_depth_mask\.bin)",
		std::regex::icase
	);
	std::smatch match;

	if (std::regex_match(filename, match, pattern)) {
		return DataPoint(match[3] == "indoors", match[1], match[2], match[4]);
	}
	return std::nullopt;
}

struct DatasetPointPaths {
	std::filesystem::path image_filepath;
	std::filesystem::path depth_filepath;
	std::filesystem::path depth_mask_filepath;
};

struct EvaluateResult {
	/// [relative0, absolute0, relative1, absolute1]
	std::vector<float> relative_absolute_pairs;
};

constexpr size_t INPUT_WIDTH = 256;
constexpr size_t INPUT_HEIGHT = 256;
constexpr size_t DATASET_WIDTH = 1024;
constexpr size_t DATASET_HEIGHT = 768;
constexpr float DATASET_MIN = 0.6f;
constexpr float DATASET_MAX = 350.f;

static tl::expected<EvaluateResult, std::string> evaluate(
	DepthModel& depth_model,
	std::span<float> image_rgb,
	std::span<float> metric_depth,
	std::span<float> depth_mask
) {
	size_t pixel_count = image_rgb.size() / 3;
	if (pixel_count != INPUT_WIDTH * INPUT_HEIGHT) {
		return tl::unexpected_fmt(
			"Invalid image size of {} instead of {}", pixel_count,
			INPUT_WIDTH * INPUT_HEIGHT
		);
	}

	if (metric_depth.size() != DATASET_WIDTH * DATASET_HEIGHT) {
		return tl::unexpected_fmt(
			"Invalid metric depth image size of {} instead of {}",
			metric_depth.size(), DATASET_WIDTH * DATASET_HEIGHT
		);
	}
	if (depth_mask.size() != DATASET_WIDTH * DATASET_HEIGHT) {
		return tl::unexpected_fmt(
			"Invalid depth mask image size of {} instead of {}",
			depth_mask.size(), DATASET_WIDTH * DATASET_HEIGHT
		);
	}

	EvaluateResult result;
	result.relative_absolute_pairs.reserve(pixel_count * 2);

	std::vector<float> depth_estimation(pixel_count);

	const auto status =
		depth_model.run(image_rgb, std::span<float>(depth_estimation));
	if (!status.has_value())
		return tl::unexpected(status.error());

	float mean_squared_metric_estimation_error = 0.f;
	size_t mean_squared_metric_estimation_error_count = 0;
	bool ignore_error = false;
	for (size_t y = 0; y < INPUT_HEIGHT; ++y) {
		for (size_t x = 0; x < INPUT_WIDTH; ++x) {
			size_t input_image_index = (y * INPUT_WIDTH) + x;
			float relative_x =
				static_cast<float>(x) / static_cast<float>(INPUT_WIDTH);
			float relative_y =
				static_cast<float>(y) / static_cast<float>(INPUT_HEIGHT);
			size_t dataset_image_index =
				(static_cast<size_t>(relative_y * DATASET_HEIGHT) *
				 DATASET_WIDTH) +
				(static_cast<size_t>(relative_x * DATASET_WIDTH));

			if (depth_mask[dataset_image_index] == 0.f)
				continue;

			float absolute = metric_depth[dataset_image_index];
			if (absolute < DATASET_MIN || absolute > DATASET_MAX)
				continue;

			// ignore probably outdoor scenes
			if (absolute > 7.5f || absolute < -2.5f)
				ignore_error = true;

			float relative = depth_estimation[input_image_index];
			result.relative_absolute_pairs.push_back(relative);
			result.relative_absolute_pairs.push_back(absolute);

			// indoors coeffs: 4.05221043e-13 -5.11228601e-10  1.23966749e-06
			// -3.01295207e-03 3.46789584e+00
			float relative2 = relative * relative;
			float relative3 = relative2 * relative;
			float relative4 = relative3 * relative;
			float metric_estimation =
				(4.05221043e-13f * relative4) - (5.11228601e-10f * relative3) +
				(1.23966749e-06f * relative2) - (3.01295207e-03f * relative) +
				3.46789584e+00f;

			float error = abs(absolute - metric_estimation);
			mean_squared_metric_estimation_error += error * error;
			mean_squared_metric_estimation_error_count++;
		}
	}

	mean_squared_metric_estimation_error /=
		static_cast<float>(mean_squared_metric_estimation_error_count);

	if (!ignore_error) {
		println_fmt(
			"mean_squared_metric_estimation_error (indoors): {}",
			mean_squared_metric_estimation_error
		);
	}

	return result;
}

static tl::expected<std::chrono::milliseconds, std::string> evaluate_set(
	DepthModel& depth_model,
	const DatasetPointPaths& dataset_point_paths,
	const std::filesystem::path& evaluation_output_filepath
) {
	const auto start = std::chrono::high_resolution_clock::now();

	auto image_result =
		read_binary_file<float>(dataset_point_paths.image_filepath);
	if (!image_result.has_value())
		return tl::unexpected(image_result.error());

	std::vector<float>& image = image_result.value();
	if (image.size() != INPUT_WIDTH * INPUT_HEIGHT * 3) {
		return tl::unexpected_fmt(
			"invalid image size of {} pixels, expected {}x{}={} pixels",
			image.size() / 3, INPUT_WIDTH, INPUT_HEIGHT,
			INPUT_WIDTH * INPUT_HEIGHT * 3
		);
	}

	auto depth_result =
		read_binary_file<float>(dataset_point_paths.depth_filepath);
	if (!depth_result.has_value())
		return tl::unexpected(depth_result.error());

	std::vector<float>& depth = depth_result.value();

	auto depth_mask_result =
		read_binary_file<float>(dataset_point_paths.depth_mask_filepath);
	if (!depth_mask_result.has_value())
		return tl::unexpected(depth_mask_result.error());

	std::vector<float>& depth_mask = depth_mask_result.value();

	const auto result = evaluate(
		depth_model, std::span<float>(image), std::span<float>(depth),
		std::span<float>(depth_mask)
	);
	if (!result.has_value())
		return tl::unexpected(result.error());

	const auto save_result = save_evaluation_result_file(
		evaluation_output_filepath,
		std::span<const float>(result.value().relative_absolute_pairs)
	);
	if (!save_result.has_value())
		return tl::unexpected(save_result.error());

	return std::chrono::duration_cast<std::chrono::milliseconds>(
		std::chrono::high_resolution_clock::now() - start
	);
}

static std::unordered_map<DataPoint, DatasetPointPaths>
scan_dataset(const std::filesystem::path& dataset_directory) {
	std::unordered_map<DataPoint, std::filesystem::path> image_filepaths;
	std::unordered_map<DataPoint, std::filesystem::path> depth_filepaths;
	std::unordered_map<DataPoint, std::filesystem::path> depth_mask_filepaths;

	for (const auto& entry :
		 std::filesystem::recursive_directory_iterator(dataset_directory)) {
		if (!entry.is_regular_file())
			continue;

		const auto& filepath = entry.path();
		const auto filename = filepath.filename();
		const std::optional<DataPoint> image_data_point =
			match_image_file(filename);
		if (image_data_point) {
			image_filepaths[*image_data_point] = filepath;
			continue;
		}
		const std::optional<DataPoint> depth_data_point =
			match_depth_file(filename);
		if (depth_data_point) {
			depth_filepaths[*depth_data_point] = filepath;
			continue;
		}

		const std::optional<DataPoint> depth_mask_data_point =
			match_depth_mask_file(filename);
		if (depth_mask_data_point) {
			depth_mask_filepaths[*depth_mask_data_point] = filepath;
		} else if (filename.extension() == ".bin") {
			println_fmt("(Skipping file {})", filepath.string());
		}
	}

	std::unordered_map<DataPoint, DatasetPointPaths> dataset_paths;
	for (const auto& [data_point, image_path] : image_filepaths) {
		if (depth_filepaths.contains(data_point)) {
			if (depth_mask_filepaths.contains(data_point)) {
				dataset_paths[data_point] = DatasetPointPaths(
					image_path, depth_filepaths.at(data_point),
					depth_mask_filepaths.at(data_point)
				);
			} else {
				println_fmt(
					"(Skipping {} with no depth mask)", data_point.to_string()
				);
			}
		} else {
			println_fmt("(Skipping {} with no depth)", data_point.to_string());
		}
	}
	for (const auto& [depth_info, depth_path] : depth_filepaths) {
		if (!dataset_paths.contains(depth_info)) {
			println_fmt(
				"(Skipping {} with no image or depth_mask)",
				depth_info.to_string()
			);
		}
	}
	for (const auto& [depth_info, depth_path] : depth_mask_filepaths) {
		if (!dataset_paths.contains(depth_info)) {
			println_fmt(
				"(Skipping {} with no image or depth)", depth_info.to_string()
			);
		}
	}
	return dataset_paths;
}