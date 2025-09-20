#include "EyeAICore/DepthModel.hpp"
#include "EyeAICore/utils/MutexGuard.hpp"
#include "EyeAICore/utils/Profiling.hpp"
#include "datasets/diode_dataset.hpp"
#include "datasets/sun_rgbd_dataset.hpp"
#include "npy.hpp"
#include "utils.hpp"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <span>

/// max number of threads for evaluation, tested such that the drive is now the
/// actual bottleneck
constexpr size_t MAX_THREAD_COUNT = 6;

int main(const int argc, const char* argv[]) {
	const auto start = std::chrono::high_resolution_clock::now();

	if (argc != 5) {
		println_error_fmt(
			"Usage: EvaluateDataset <diode or sun_rgbd> <midas.tflite> "
			"<dataset_directory> <evaluation_output_directory>"
		);
		return 1;
	}

	std::span<const char*> args(argv, argc);

	const std::filesystem::path temp_dir =
		std::filesystem::temp_directory_path();
	const std::string_view dataset_type = args[1];
	const std::filesystem::path midas_model_path = args[2];
	const std::filesystem::path dataset_directory = args[3];
	const std::filesystem::path prepared_output_directory = args[4];

	const auto midas_model_last_modified =
		std::filesystem::last_write_time(midas_model_path);

	const std::filesystem::path gpu_delegate_serialization_dir =
		temp_dir / "EyeAICore/gpu_delegate_cache";
	std::filesystem::create_directories(gpu_delegate_serialization_dir);
	const std::string midas_model_token = std::format(
		"{}_{}", midas_model_path.filename().string(), midas_model_last_modified
	);

	// not used, since mobile phones with qualcomm npu's are not preparing datasets
	const std::filesystem::path npu_delegate_dir = temp_dir / "qnn_npu_delegate";
	std::filesystem::create_directories(npu_delegate_dir);

	const size_t thread_count = std::min(
		MAX_THREAD_COUNT,
		static_cast<size_t>(std::thread::hardware_concurrency())
	);

	std::cout << "\n=== Initializing TFLite Runtime ===\n\n";

	auto model_data_result = read_binary_file(midas_model_path);
	if (!model_data_result.has_value()) {
		println_error_fmt(
			"Failed to read model file: {}", model_data_result.error()
		);
		return 1;
	}
	auto& model_data = model_data_result.value();

	TfLiteLogWarningCallback tflite_log_warning_callback = [](std::string msg) {
		println_fmt("[TfLite Warning] {}", msg);
	};
	TfLiteLogErrorCallback tflite_log_error_callback = [](std::string msg) {
		println_error_fmt("[TfLite Error] {}", msg);
	};

	std::unique_ptr<RGBDDataset> dataset;
	if (dataset_type == "diode")
		dataset = std::make_unique<DiodeDataset>();
	else if (dataset_type == "sun_rgbd")
		dataset = std::make_unique<SUN_RGBD_Dataset>();
	else {
		println_error_fmt("unknown dataset type: {}", dataset_type);
		return 1;
	}

	std::cout << "\n=== Searching Dataset for scans ===\n\n";

	const auto diode_scan = dataset->scan(dataset_directory);

	std::cout << "\n=== Preparing Dataset ===\n\n";

	std::filesystem::create_directories(prepared_output_directory);

	MutexGuard<
		std::array<size_t, RAW_RELATIVE_DEPTH_VALUE_DISTRIBUTION_BIN_COUNT>>
		raw_relative_depth_value_distribution{};

	std::atomic_size_t current_scan_index = 0;

	{
		const auto depth_model_thread_context_generator =
			[&]() -> std::unique_ptr<DepthModel> {
			PROFILE_DEPTH_SCOPE("Create Depth Model ThreadContextGenerator")

			auto model_data_clone = model_data;

			auto result = DepthModel::create(
				std::move(model_data_clone),
				gpu_delegate_serialization_dir.string(), midas_model_token,
				tflite_log_warning_callback, tflite_log_error_callback, false
			);

			if (result) {
				return std::move(result.value());
			}
			println_error_fmt(
				"Failed to create depth model, aborting: {}",
				result.error().to_string()
			);
			exit(1);
		};

		ThreadPool<std::unique_ptr<DepthModel>> pool(
			depth_model_thread_context_generator, thread_count
		);

		size_t scan_size = diode_scan.size();
		std::atomic_size_t current_scan_index = 0;
		for (const auto& data_point : diode_scan) {
			pool.enqueue([&,
						  scan_size](std::unique_ptr<DepthModel>& depth_model) {
				PROFILE_DEPTH_SCOPE("Evaluate datapoint ThreadPool Task")

				auto preparation_result = prepare_datapoint(
					*depth_model, *data_point, prepared_output_directory,
					current_scan_index.load()
				);

				if (!preparation_result) {
					println_error_fmt(
						"   Failed to evaluate datapoint: {}, skipping!",
						preparation_result.error()
					);
					return;
				}

				// Accumulate raw relative depth value distribution
				{
					auto raw_relative_depth_value_distribution_scope =
						raw_relative_depth_value_distribution.lock();

					for (size_t i = 0;
						 i <
						 raw_relative_depth_value_distribution_scope->size();
						 ++i) {
						(*raw_relative_depth_value_distribution_scope).at(i) +=
							preparation_result
								->raw_relative_depth_value_distribution.at(i);
					}
				}

				const float scan_percentage =
					static_cast<float>(current_scan_index + 1) /
					static_cast<float>(scan_size);
				println_fmt(
					"=== Scan [{}/{} {}%] evaluation took {} ms ===\n",
					current_scan_index + 1, scan_size,
					static_cast<int>(scan_percentage * 100.f),
					preparation_result->duration.count()
				);
				current_scan_index++;
			});
		}
	}

	// Save raw relative depth biased samples
	{
		auto raw_relative_depth_samples = generate_raw_relative_depth_samples(
			*(raw_relative_depth_value_distribution.const_lock())
		);

		std::ranges::sort(raw_relative_depth_samples);

		npy::write_npy(
			prepared_output_directory / "raw_relative_depth_samples.npy",
			npy::npy_data_ptr<float>(
				raw_relative_depth_samples.data(),
				{raw_relative_depth_samples.size()}
			)
		);
	}

	const auto total_duration =
		std::chrono::duration_cast<std::chrono::seconds>(
			std::chrono::high_resolution_clock::now() - start
		);
	println_fmt(
		"==========================\nAll {} images finished! Total time "
		"taken: {} s",
		diode_scan.size(), total_duration.count()
	);

	const size_t expected_image_count = dataset->expected_image_count();
	if (diode_scan.size() != expected_image_count) {
		println_error_fmt(
			"Warning: Searching the dataset found {} scanned images, but {} "
			"were expected!",
			diode_scan.size(), expected_image_count
		);
	}
}
