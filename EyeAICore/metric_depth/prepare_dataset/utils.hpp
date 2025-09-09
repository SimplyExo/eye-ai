#pragma once

#include "EyeAICore/DepthModel.hpp"
#include "EyeAICore/TensorBuffer.hpp"
#include "datasets/dataset.hpp"
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <functional>
#include <iostream>
#include <npy.hpp>
#include <queue>
#include <span>
#include <stb_image.h>
#include <stb_image_resize2.h>
#include <tl/expected.hpp>

[[nodiscard]] tl::expected<std::vector<int8_t>, std::string>
read_binary_file(const std::filesystem::path& filepath);

template<typename... Args>
void println_fmt(const std::format_string<Args...> fmt, Args&&... args) {
	const std::string formatted =
		std::vformat(fmt.get(), std::make_format_args(args...));

	std::cout << formatted << '\n';
}

template<typename... Args>
void println_error_fmt(const std::format_string<Args...> fmt, Args&&... args) {
	const std::string formatted =
		std::vformat(fmt.get(), std::make_format_args(args...));

	std::cerr << formatted << '\n';
}

struct PreparedImage {
	FloatTensorBuffer<FloatTensorFormat::Rel2AbsDepthInput> rgbd;
	FloatTensorBuffer<FloatTensorFormat::Rel2AbsDepthCoefficientOutput> coeffs;
	std::vector<std::pair<float, float>> relative_absolute_depth_pairs;
};

constexpr static size_t RAW_RELATIVE_DEPTH_VALUE_DISTRIBUTION_BIN_COUNT = 1000;
constexpr static float RAW_RELATIVE_DEPTH_VALUE_DISTRIBUTION_BIN_WIDTH =
	1500.0f / (float)RAW_RELATIVE_DEPTH_VALUE_DISTRIBUTION_BIN_COUNT;
constexpr static size_t RAW_RELATIVE_DEPTH_SAMPLE_COUNT = 100;

struct PreparationResult {
	std::chrono::milliseconds duration;
	std::array<size_t, RAW_RELATIVE_DEPTH_VALUE_DISTRIBUTION_BIN_COUNT>
		raw_relative_depth_value_distribution;
};

/// rgb image of rgbd_image must match depth_input dimensions, depth image of
/// rgbd_image will be resized to match depth_input dimensions
tl::expected<PreparedImage, std::string> prepare(
	DepthModel& depth_model,
	size_t depth_input_width,
	size_t depth_input_height,
	const RGBDImage& rgbd_image,
	std::array<size_t, RAW_RELATIVE_DEPTH_VALUE_DISTRIBUTION_BIN_COUNT>&
		out_raw_relative_depth_value_distribution
);

tl::expected<PreparationResult, std::string> prepare_datapoint(
	DepthModel& depth_model,
	const RGBDDataPoint& datapoint,
	const std::filesystem::path& prepared_output_directory,
	size_t datapoint_index
);

FloatTensorBuffer<FloatTensorFormat::Rel2AbsDepthCoefficientOutput>
find_coeffs(std::span<std::pair<float, float>> relative_absolute_pairs);

std::array<float, RAW_RELATIVE_DEPTH_SAMPLE_COUNT>
generate_raw_relative_depth_samples(
	std::span<const size_t, RAW_RELATIVE_DEPTH_VALUE_DISTRIBUTION_BIN_COUNT>
		raw_relative_depth_value_distribution
);

tl::expected<FloatTensorBuffer<FloatTensorFormat::ImageRGB255>, std::string>
load_rgb_image_file(
	const std::filesystem::path& filepath,
	size_t target_width,
	size_t target_height
);

tl::expected<std::vector<uint16_t>, std::string>
load_16bit_greyscale_image_file(
	const std::filesystem::path& filepath,
	size_t& out_width,
	size_t& out_height
);

tl::expected<std::vector<float>, std::string>
load_npy_file(const std::filesystem::path& filepath);

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
