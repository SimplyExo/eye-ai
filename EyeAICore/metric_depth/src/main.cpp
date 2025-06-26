#include "EyeAICore/DepthModel.hpp"
#include "utils.hpp"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <span>
#include <unordered_map>

/// max number of threads for evaluation, tested such that the drive is now the
/// actual bottleneck
constexpr size_t MAX_THREAD_COUNT = 6;

int main(const int argc, const char* argv[]) {
	const auto start = std::chrono::high_resolution_clock::now();

	if (argc != 4) {
		println_error_fmt(
			"Usage: EvaluateDataset <midas.tflite> <dataset_directory> "
			"<evaluation_output_directory>"
		);
		return 1;
	}

	std::span<const char*> args(argv, argc);

	const std::filesystem::path temp_dir =
		std::filesystem::temp_directory_path();
	const std::filesystem::path midas_model_path = args[1];
	const std::filesystem::path dataset_directory = args[2];
	const std::filesystem::path evaluation_output_directory = args[3];

	const auto midas_model_last_modified =
		std::filesystem::last_write_time(midas_model_path);

	const std::filesystem::path gpu_delegate_serialization_dir =
		temp_dir / "EyeAICore/gpu_delegate_cache";
	std::filesystem::create_directories(gpu_delegate_serialization_dir);
	const std::string midas_model_token = std::format(
		"{}_{}", midas_model_path.filename().string(), midas_model_last_modified
	);

	const size_t thread_count = std::min(
		MAX_THREAD_COUNT,
		static_cast<size_t>(std::thread::hardware_concurrency())
	);

	std::cout << "\n=== Initializing TFLite Runtime ===\n\n";

	auto model_data_result = read_binary_file<int8_t>(midas_model_path);
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

	std::cout << "\n=== Searching Dataset for scans ===\n\n";

	const auto scans = search_for_scans_in_dataset(dataset_directory);

	std::cout << "\n=== Evaluating Dataset ===\n\n";

	std::filesystem::path indoors_directory =
		evaluation_output_directory / "indoors";
	std::filesystem::create_directories(indoors_directory);
	std::filesystem::path outdoors_directory =
		evaluation_output_directory / "outdoor";
	std::filesystem::create_directories(outdoors_directory);

	std::atomic_size_t total_image_count = 0;
	std::atomic_size_t current_scan_index = 0;

	{
		ThreadPool<std::unique_ptr<DepthModel>> pool(
			[&]() -> std::unique_ptr<DepthModel> {
				auto model_data_clone = model_data;

				return DepthModel::create_with_raw_output(
						   std::move(model_data_clone),
						   gpu_delegate_serialization_dir.string(),
						   midas_model_token, tflite_log_warning_callback,
						   tflite_log_error_callback
				)
					.value();
			},
			thread_count
		);

		for (const auto& [scan_id, scan_directory] : scans) {
			const auto scans_size = scans.size();

			pool.enqueue([&](std::unique_ptr<DepthModel>& depth_model) {
				const DatasetScan dataset_scan =
					search_for_images_in_scan(scan_directory);

				const auto scan_evaluation_start =
					std::chrono::high_resolution_clock::now();
				size_t image_index = 0;
				for (const auto& [datapoint, paths] : dataset_scan.paths) {
					image_index++;
					total_image_count++;

					const auto result_filepath =
						(datapoint.indoors ? indoors_directory
										   : outdoors_directory) /
						std::format(
							"{}_{}_{}_result.bin", datapoint.scene_id,
							datapoint.scan_id, datapoint.imgname
						);

					const auto result =
						evaluate_set(*depth_model, paths, result_filepath);

					if (!result.has_value()) {
						println_error_fmt(
							"   Failed with error: {}, skipping!",
							result.error()
						);
					}
				}

				const auto scan_evaluation_duration =
					std::chrono::duration_cast<std::chrono::milliseconds>(
						std::chrono::high_resolution_clock::now() -
						scan_evaluation_start
					);

				const float scan_percentage =
					static_cast<float>(current_scan_index + 1) /
					static_cast<float>(scans_size);
				println_fmt(
					"=== Scan {} [{}/{} {}%] evaluation took {} ms ===\n",
					scan_id, current_scan_index + 1, scans_size,
					static_cast<int>(scan_percentage * 100.f),
					scan_evaluation_duration.count()
				);
				current_scan_index++;
			});
		}
	}

	const auto total_duration =
		std::chrono::duration_cast<std::chrono::seconds>(
			std::chrono::high_resolution_clock::now() - start
		);
	println_fmt(
		"==========================\nAll {} images finished! Total time "
		"taken: {} s",
		total_image_count.load(), total_duration.count()
	);

	if (total_image_count != 771) {
		println_error_fmt(
			"Warning: Searching the dataset found {} scanned images, but 771 "
			"were expected!",
			total_image_count.load()
		);
	}
}
