#include "utils.hpp"
#include <chrono>
#include <filesystem>
#include <format>
#include <iostream>
#include <span>
#include <unordered_map>

int main(const int argc, const char* argv[]) {
	const auto start = std::chrono::high_resolution_clock::now();

	if (argc != 4) {
		println_error_fmt(
			"Usage: EvaluateDataset <midas.tflite> <dataset_directory> "
			"<evaluation_output_directory>\n"
		);
		return 1;
	}

	std::span<const char*> args(argv, argc);

	const std::filesystem::path midas_model_path = args[1];
	const std::filesystem::path dataset_directory = args[2];
	const std::filesystem::path evaluation_output_directory = args[3];

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

	auto runtime_result = TfLiteRuntime::create(
		std::move(model_data), GPU_DELEGATE_SERIALIZATION_DIR,
		MIDAS_MODEL_TOKEN, tflite_log_warning_callback,
		tflite_log_error_callback
	);

	if (!runtime_result.has_value()) {
		println_error_fmt(
			"Could not create TfLiteRuntime: {}", runtime_result.error()
		);
		return 1;
	}
	auto& runtime = runtime_result.value();

	std::cout << "\n=== Scanning Dataset for entries ===\n\n";

	const auto dataset_paths = scan_dataset(dataset_directory);

	std::cout << "\n=== Evaluating Dataset ===\n\n";

	std::filesystem::path indoors_directory =
		evaluation_output_directory / "indoors";
	std::filesystem::create_directories(indoors_directory);
	std::filesystem::path outdoors_directory =
		evaluation_output_directory / "outdoor";
	std::filesystem::create_directories(outdoors_directory);

	size_t i = 0;
	for (const auto& [data_point, paths] : dataset_paths) {
		const float percentage = static_cast<float>(i) /
								 static_cast<float>(dataset_paths.size() - 1);
		const std::string progress = std::format(
			"[{}/{} {}%]", i + 1, dataset_paths.size(),
			static_cast<int>(percentage * 100.f)
		);
		std::cout << progress << " === Evaluating " << data_point.to_string()
				  << " ===";
		i++;

		const auto result_filepath =
			(data_point.indoors ? indoors_directory : outdoors_directory) /
			std::format(
				"{}_{}_{}_result.csv", data_point.scene_id, data_point.scan_id,
				data_point.imgname
			);

		const auto result = evaluate_set(*runtime, paths, result_filepath);

		if (result.has_value()) {
			println_fmt("    Finished, took {} ms", result.value().count());
		} else {
			println_error_fmt(
				"   Failed with error: {}, skipping!\n", result.error()
			);
		}
	}

	const auto total_duration =
		std::chrono::duration_cast<std::chrono::seconds>(
			std::chrono::high_resolution_clock::now() - start
		);
	println_fmt(
		"\n==========================\nAll {} scans finished! Total time "
		"taken: {} s\n",
		dataset_paths.size(), total_duration.count()
	);

	if (dataset_paths.size() != 771) {
		println_error_fmt(
			"Scanning the dataset found {}, but 771 were expected!",
			dataset_paths.size()
		);
	}
}
