#pragma once

#include "EyeAICore/DepthModel.hpp"
#include "EyeAICore/utils/Errors.hpp"
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <npy.hpp>
#include <queue>
#include <regex>
#include <span>
#include <stb_image.h>
#include <stb_image_resize2.h>
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

	file.write(
		reinterpret_cast<const char*>(relative_absolute_pairs.data()),
		static_cast<std::streamsize>(relative_absolute_pairs.size_bytes())
	);

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
		R"((\d+)_(\d+)_(outdoor|indoors)_(\w+)\.png)", std::regex::icase
	);
	std::smatch match;

	if (std::regex_match(filename, match, pattern)) {
		return DataPoint(match[3] == "indoors", match[1], match[2], match[4]);
	}
	return std::nullopt;
}

static std::optional<DataPoint> match_depth_file(const std::string& filename) {
	std::regex pattern(
		R"((\d+)_(\d+)_(outdoor|indoors)_(\w+)_depth\.npy)", std::regex::icase
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
		R"((\d+)_(\d+)_(outdoor|indoors)_(\w+)_depth_mask\.npy)",
		std::regex::icase
	);
	std::smatch match;

	if (std::regex_match(filename, match, pattern)) {
		return DataPoint(match[3] == "indoors", match[1], match[2], match[4]);
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

			float relative = depth_estimation[input_image_index];
			result.relative_absolute_pairs.push_back(relative);
			result.relative_absolute_pairs.push_back(absolute);
		}
	}

	return result;
}

static tl::expected<std::vector<float>, std::string>
load_image_file(const std::filesystem::path& filepath) {
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

	const size_t target_width = INPUT_WIDTH;
	const size_t target_height = INPUT_HEIGHT;
	std::vector<float> resized_image(target_width * target_height * STBI_rgb);

	stbir_resize_float_linear(
		data, width, height, 0, resized_image.data(), target_width,
		target_height, 0, STBIR_RGB
	);

	stbi_image_free(data);

	return resized_image;
}

static tl::expected<std::vector<float>, std::string>
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

static tl::expected<std::chrono::milliseconds, std::string> evaluate_set(
	DepthModel& depth_model,
	const DatasetPointPaths& dataset_point_paths,
	const std::filesystem::path& evaluation_output_filepath
) {
	const auto start = std::chrono::high_resolution_clock::now();

	auto image_result = load_image_file(dataset_point_paths.image_filepath);
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

	auto depth_result = load_npy_file(dataset_point_paths.depth_filepath);
	if (!depth_result.has_value())
		return tl::unexpected(depth_result.error());

	std::vector<float>& depth = depth_result.value();

	auto depth_mask_result =
		load_npy_file(dataset_point_paths.depth_mask_filepath);
	if (!depth_mask_result.has_value())
		return tl::unexpected(depth_mask_result.error());

	std::vector<float>& depth_mask = depth_mask_result.value();

	const auto result = evaluate(depth_model, image, depth, depth_mask);
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

struct DatasetScan {
	std::filesystem::path directory;
	std::unordered_map<DataPoint, DatasetPointPaths> paths;
};

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

static DatasetScan
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

/// Simple thread pool, with a context for each thread, that can be referenced
/// by the enqueued tasks
template<typename Context>
class ThreadPool {
  public:
	using Task = std::function<void(Context&)>;

	explicit ThreadPool(
		std::function<Context()> thread_context_generator,
		size_t num_threads
	) {
		for (size_t i = 0; i < num_threads; ++i) {

			workers.emplace_back([this, thread_context_generator] {
				Context thread_context = thread_context_generator();

				while (true) {
					Task task;
					{
						std::unique_lock<std::mutex> lock(this->queue_mutex);
						this->condition.wait(lock, [this] {
							return this->stop || !this->tasks.empty();
						});
						if (this->stop && this->tasks.empty())
							return;
						task = std::move(this->tasks.front());
						this->tasks.pop();
					}
					task(thread_context);
				}
			});
		}
	}

	~ThreadPool() {
		{
			std::unique_lock<std::mutex> lock(queue_mutex);
			stop = true;
		}
		condition.notify_all();
		for (std::thread& worker : workers) {
			worker.join();
		}
	}

	ThreadPool& operator=(const ThreadPool&) = delete;
	ThreadPool& operator=(ThreadPool&&) = delete;
	ThreadPool(const ThreadPool&) = delete;
	ThreadPool(ThreadPool&&) = delete;

	void enqueue(Task&& task) {
		{
			std::unique_lock<std::mutex> lock(queue_mutex);
			tasks.emplace(std::move(task));
		}
		condition.notify_one();
	}

  private:
	std::vector<std::thread> workers;
	std::queue<Task> tasks;
	std::mutex queue_mutex;
	std::condition_variable condition;
	bool stop = false;
};
