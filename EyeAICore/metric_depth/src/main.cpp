#include "utils.hpp"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <span>
#include <unordered_map>

int main(const int argc, const char* argv[]) {
	const auto start = std::chrono::high_resolution_clock::now();

	if (argc != 5) {
		println_error_fmt(
			"Usage: EvaluateDataset <midas.tflite> <prepare_dataset.py> "
			"<dataset_directory> "
			"<evaluation_output_directory>"
		);
		return 1;
	}

	std::span<const char*> args(argv, argc);

	const std::filesystem::path midas_model_path = args[1];
	const std::filesystem::path prepare_dataset_python_path = args[2];
	const std::filesystem::path dataset_directory = args[3];
	const std::filesystem::path evaluation_output_directory = args[4];
	const std::filesystem::path prepared_dataset_scan_directory =
		"/tmp/EyeAICore/prepared_dataset_scan";
	std::filesystem::remove_all(prepared_dataset_scan_directory);

	constexpr const char* GPU_DELEGATE_SERIALIZATION_DIR =
		"/tmp/EyeAICore/gpu_delegate_cache";
	std::filesystem::create_directories(GPU_DELEGATE_SERIALIZATION_DIR);
	constexpr const char* MIDAS_MODEL_TOKEN = "ijustmadethistokenup";

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

	auto depth_model_result = DepthModel::create_with_raw_output(
		std::move(model_data), GPU_DELEGATE_SERIALIZATION_DIR,
		MIDAS_MODEL_TOKEN, tflite_log_warning_callback,
		tflite_log_error_callback
	);

	if (!depth_model_result.has_value()) {
		println_error_fmt(
			"Could not create depth model: {}", depth_model_result.error()
		);
		return 1;
	}
	auto& depth_model = depth_model_result.value();

	std::cout << "\n=== Searching Dataset for scans ===\n\n";

	const auto scans = search_for_scans_in_dataset(dataset_directory);

	std::cout << "\n=== Evaluating Dataset ===\n\n";

	std::filesystem::path indoors_directory =
		evaluation_output_directory / "indoors";
	std::filesystem::create_directories(indoors_directory);
	std::filesystem::path outdoors_directory =
		evaluation_output_directory / "outdoor";
	std::filesystem::create_directories(outdoors_directory);

	size_t total_image_count = 0;
	size_t current_scan_index = 0;
	for (const auto& [scan_id, scan_directory] : scans) {
		const float scan_percentage =
			static_cast<float>(current_scan_index + 1) /
			static_cast<float>(scans.size());
		println_fmt(
			"=== Scan {} [{}/{} {}%] ===", scan_id, current_scan_index + 1,
			scans.size(), scan_percentage
		);
		current_scan_index++;

		const int preparation_exit_code = prepare_dataset_scan(
			prepare_dataset_python_path, scan_directory,
			prepared_dataset_scan_directory
		);

		if (preparation_exit_code != EXIT_SUCCESS) {
			println_error_fmt(
				"Failed to prepare dataset scan using python script {}, exited "
				"with code {}",
				prepare_dataset_python_path.string(), preparation_exit_code
			);
			continue;
		}

		std::string progress_bar;

		const DatasetScan dataset_scan =
			search_for_images_in_prepared_scan(prepared_dataset_scan_directory);

		const auto scan_evaluation_start =
			std::chrono::high_resolution_clock::now();
		size_t image_index = 0;
		for (const auto& [datapoint, paths] : dataset_scan.paths) {
			const float percentage =
				static_cast<float>(image_index + 1) /
				static_cast<float>(dataset_scan.paths.size());
			progress_bar = std::format(
				"[{}/{} {}%] Evaluating {}", image_index + 1,
				dataset_scan.paths.size(), static_cast<int>(percentage * 100.f),
				datapoint.to_string()
			);
			std::cout << '\r' << progress_bar;
			std::cout.flush();
			image_index++;
			total_image_count++;

			const auto result_filepath =
				(datapoint.indoors ? indoors_directory : outdoors_directory) /
				std::format(
					"{}_{}_{}_result.csv", datapoint.scene_id,
					datapoint.scan_id, datapoint.imgname
				);

			const auto result =
				evaluate_set(*depth_model, paths, result_filepath);

			if (result.has_value()) {
				progress_bar +=
					std::format(", took {} ms", result.value().count());
				std::cout << '\r' << progress_bar;
				std::cout.flush();
			} else {
				progress_bar.clear();
				println_error_fmt(
					"   Failed with error: {}, skipping!", result.error()
				);
			}
		}

		const auto scan_evaluation_duration =
			std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::high_resolution_clock::now() -
				scan_evaluation_start
			);

		if (!progress_bar.empty()) {
			for (char& c : progress_bar)
				c = ' ';
			std::cout << '\r' << progress_bar;
			std::cout.flush();
		}
		std::cout << "\rScan evaluation took "
				  << scan_evaluation_duration.count() << " ms\n\n";
		std::cout.flush();

		std::filesystem::remove_all(prepared_dataset_scan_directory);
	}

	const auto total_duration =
		std::chrono::duration_cast<std::chrono::seconds>(
			std::chrono::high_resolution_clock::now() - start
		);
	println_fmt(
		"==========================\nAll {} images finished! Total time "
		"taken: {} s",
		total_image_count, total_duration.count()
	);

	if (total_image_count != 771) {
		println_error_fmt(
			"Warning: Searching the dataset found {} scanned images, but 771 "
			"were expected!",
			total_image_count
		);
	}
}
