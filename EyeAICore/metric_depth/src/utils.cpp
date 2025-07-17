#include "utils.hpp"
#include "EyeAICore/utils/Errors.hpp"
#include <fstream>
#include <regex>

tl::expected<std::vector<int8_t>, std::string>
read_binary_file(const std::filesystem::path& filepath) {
	std::ifstream file(filepath, std::ios::binary | std::ios::ate);

	if (!file.is_open())
		return tl::unexpected_fmt("Failed to open file: {}", filepath.string());

	std::streamsize binary_size = file.tellg();
	file.seekg(0, std::ios::beg);

	std::vector<int8_t> buffer(binary_size);

	if (!file.read(reinterpret_cast<char*>(buffer.data()), binary_size))
		return tl::unexpected_fmt("Failed to read file: {}", filepath.string());

	return buffer;
}

tl::expected<void, std::string> save_evaluation_result_file(
	const std::filesystem::path& filepath,
	std::span<const float> relative_absolute_pairs
) {
	std::filesystem::create_directories(filepath.parent_path());
	std::ofstream file(filepath);
	if (!file.is_open())
		return tl::unexpected_fmt("Failed to open file: {}", filepath.string());

	file.write(
		reinterpret_cast<const char*>(relative_absolute_pairs.data()),
		static_cast<std::streamsize>(relative_absolute_pairs.size_bytes())
	);

	file.flush();

	return {};
}

std::string DataPoint::to_string() const noexcept {
	return std::format(
		"{} scene {}, scan {}, image {}", indoors ? "indoors" : "outdoor",
		scene_id, scan_id, imgname
	);
}

std::size_t
std::hash<DataPoint>::operator()(const DataPoint& dp) const noexcept {
	return std::hash<bool>{}(dp.indoors) ^
		   std::hash<std::string>{}(dp.scene_id) ^
		   std::hash<std::string>{}(dp.scan_id) ^
		   std::hash<std::string>{}(dp.imgname);
}

std::optional<DataPoint> match_image_file(const std::string& filename) {
	std::regex pattern(
		R"((\d+)_(\d+)_(outdoor|indoors)_(\w+)\.png)", std::regex::icase
	);
	std::smatch match;

	if (std::regex_match(filename, match, pattern)) {
		return DataPoint(match[3] == "indoors", match[1], match[2], match[4]);
	}
	return std::nullopt;
}

std::optional<DataPoint> match_depth_file(const std::string& filename) {
	std::regex pattern(
		R"((\d+)_(\d+)_(outdoor|indoors)_(\w+)_depth\.npy)", std::regex::icase
	);
	std::smatch match;

	if (std::regex_match(filename, match, pattern)) {
		return DataPoint(match[3] == "indoors", match[1], match[2], match[4]);
	}
	return std::nullopt;
}

std::optional<DataPoint> match_depth_mask_file(const std::string& filename) {
	std::regex pattern(
		R"((\d+)_(\d+)_(outdoor|indoors)_(\w+)_depth_mask\.npy)",
		std::regex::icase
	);
	std::smatch match;

	if (std::regex_match(filename, match, pattern)) {
		return DataPoint(match[3] == "indoors", match[1], match[2], match[4]);
	}
	return std::nullopt;
}

std::optional<std::string> match_scan_directory(const std::string& directory) {
	std::regex pattern(R"(scan_(\d+))", std::regex::icase);
	std::smatch match;

	if (std::regex_match(directory, match, pattern)) {
		return match[1];
	}
	return std::nullopt;
}

std::string format_span(std::span<const int> s) {
	if (s.empty())
		return "[]";
	std::string result = "[";
	for (size_t i = 0; i < s.size(); ++i) {
		result += std::to_string(s[i]);
		if (i < s.size() - 1)
			result += ", ";
	}
	result += "]";
	return result;
}

tl::expected<EvaluateResult, std::string> evaluate(
	DepthModel& depth_model,
	size_t depth_input_width,
	size_t depth_input_height,
	std::span<float> image_rgb,
	std::span<float> metric_depth,
	std::span<float> depth_mask
) {
	size_t pixel_count = image_rgb.size() / 3;
	if (pixel_count != depth_input_width * depth_input_height) {
		return tl::unexpected_fmt(
			"Invalid image size of {} instead of {}", pixel_count,
			depth_input_width * depth_input_height
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

	if (const auto error =
			depth_model.run(image_rgb, std::span<float>(depth_estimation))) {
		return tl::unexpected(error->to_string());
	}

	for (size_t y = 0; y < depth_input_height; ++y) {
		for (size_t x = 0; x < depth_input_width; ++x) {
			size_t input_image_index = (y * depth_input_width) + x;
			float relative_x =
				static_cast<float>(x) / static_cast<float>(depth_input_width);
			float relative_y =
				static_cast<float>(y) / static_cast<float>(depth_input_height);
			size_t dataset_image_index =
				(static_cast<size_t>(relative_y * DATASET_HEIGHT) *
				 DATASET_WIDTH) +
				(static_cast<size_t>(relative_x * DATASET_WIDTH));

			if (depth_mask[dataset_image_index] == 0.f)
				continue;

			float absolute = metric_depth[dataset_image_index];
			if (absolute < DATASET_MIN || absolute > DATASET_MAX)
				continue;

			float relative = depth_estimation[input_image_index];
			result.relative_absolute_pairs.push_back(relative);
			result.relative_absolute_pairs.push_back(absolute);
		}
	}

	return result;
}

tl::expected<std::vector<float>, std::string> load_image_file(
	const std::filesystem::path& filepath,
	size_t target_width,
	size_t target_height
) {
	const std::string filepath_str = filepath.string();
	int width = 0;
	int height = 0;
	int channels = 3;
	float* data =
		stbi_loadf(filepath_str.c_str(), &width, &height, &channels, STBI_rgb);
	if (channels != STBI_rgb) {
		return tl::unexpected_fmt(
			"invalid channels other than RGB in image file {}", filepath_str
		);
	}
	if (data == nullptr) {
		return tl::unexpected_fmt("failed to load image file {}", filepath_str);
	}

	std::vector<float> resized_image(target_width * target_height * STBI_rgb);

	stbir_resize_float_linear(
		data, width, height, 0, resized_image.data(),
		static_cast<int>(target_width), static_cast<int>(target_height), 0,
		STBIR_RGB
	);

	stbi_image_free(data);

	return resized_image;
}

tl::expected<std::vector<float>, std::string>
load_npy_file(const std::filesystem::path& filepath) {
	// first try loading as float
	try {
		const auto npy_data = npy::read_npy<float>(filepath);
		return npy_data.data;
	} catch (const std::exception& e) {
		// then as double -> float
		try {
			const auto npy_data = npy::read_npy<double>(filepath);
			std::vector<float> values(npy_data.data.size());
			for (size_t i = 0; i < values.size(); ++i)
				values[i] = static_cast<float>(npy_data.data[i]);
			return values;
		} catch (const std::exception& e) {
			return tl::unexpected_fmt(
				"failed to load npy file {}: {}", filepath.string(), e.what()
			);
		}
	}
}

tl::expected<std::chrono::milliseconds, std::string> evaluate_set(
	DepthModel& depth_model,
	const DatasetPointPaths& dataset_point_paths,
	const std::filesystem::path& evaluation_output_filepath
) {
	const auto start = std::chrono::high_resolution_clock::now();

	const std::span<const int> input_shape = depth_model.get_input_shape();
	if (input_shape.size() != 4) {
		return tl::unexpected_fmt(
			"invalid input shape dimensions, expected 4 but has {}",
			input_shape.size()
		);
	}
	if (input_shape[0] != 1) {
		return tl::unexpected_fmt(
			"invalid batch size, expected 1 but has {}", input_shape[0]
		);
	}
	if (input_shape[3] != 3) {
		return tl::unexpected_fmt(
			"invalid channel size, expected 3 (r,g,b) but has {}",
			input_shape[3]
		);
	}
	auto depth_input_width = static_cast<size_t>(input_shape[2]);
	auto depth_input_height = static_cast<size_t>(input_shape[1]);

	const std::span<const int> output_shape = depth_model.get_output_shape();
	const std::array<int, 4> expected_output_shape{
		1, input_shape[1], input_shape[2], 1
	};
	if (!std::ranges::equal(output_shape, expected_output_shape)) {
		return tl::unexpected_fmt(
			"invalid output shape, expected {} but has {}",
			format_span(expected_output_shape), format_span(output_shape)
		);
	}

	auto image_result = load_image_file(
		dataset_point_paths.image_filepath, depth_input_width,
		depth_input_height
	);
	if (!image_result.has_value())
		return tl::unexpected(image_result.error());

	std::vector<float>& image = image_result.value();
	if (image.size() != depth_input_width * depth_input_height * 3) {
		return tl::unexpected_fmt(
			"invalid image size of {} pixels, expected {}x{}={} pixels",
			image.size() / 3, depth_input_width, depth_input_height,
			depth_input_width * depth_input_height * 3
		);
	}

	auto depth_result = load_npy_file(dataset_point_paths.depth_filepath);
	if (!depth_result.has_value())
		return tl::unexpected(depth_result.error());

	std::vector<float>& depth = depth_result.value();

	auto depth_mask_result =
		load_npy_file(dataset_point_paths.depth_mask_filepath);
	if (!depth_mask_result.has_value())
		return tl::unexpected(depth_mask_result.error());

	std::vector<float>& depth_mask = depth_mask_result.value();

	const auto result = evaluate(
		depth_model, depth_input_width, depth_input_height, image, depth,
		depth_mask
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

std::unordered_map<std::string, std::filesystem::path>
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

DatasetScan
search_for_images_in_scan(const std::filesystem::path& scan_directory) {
	std::unordered_map<DataPoint, std::filesystem::path> image_filepaths;
	std::unordered_map<DataPoint, std::filesystem::path> depth_filepaths;
	std::unordered_map<DataPoint, std::filesystem::path> depth_mask_filepaths;

	for (const auto& entry :
		 std::filesystem::directory_iterator(scan_directory)) {

		const auto& filepath = entry.path();
		const auto filename = filepath.filename();
		if (!entry.is_regular_file())
			continue;

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

	return DatasetScan(scan_directory, dataset_paths);
}