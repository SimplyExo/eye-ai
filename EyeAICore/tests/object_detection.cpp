#include "EyeAICore/YoloModel.hpp"
#include "utils.hpp"
#include <filesystem>
#include <iostream>
#include <string>

constexpr const char* YOLO_MODEL_TOKEN = "ijustmadethistokenup_yolo";
constexpr size_t INPUT_WIDTH = 640;
constexpr size_t INPUT_HEIGHT = 640;
constexpr size_t OUTPUT_BUFFER_SIZE = 84 * 8400;

TEST(ObjectDetection, CorrectOutput) {
	const auto gpu_delegate_serialization_dir =
		std::filesystem::temp_directory_path() / "EyeAICore" /
		"gpu_delegate_cache";

	std::filesystem::create_directories(gpu_delegate_serialization_dir);

	auto model_data_result = read_model_data(
		"../../EyeAIApp/"
		"app/src/main/assets/model.tflite"
	);
	EXPECT_RESULT_HAS_VALUE(model_data_result);
	auto& model_data = model_data_result.value();

	TfLiteLogWarningCallback tflite_log_warning_callback = [](std::string msg) {
		std::cout << "[TfLite Warning] " << msg << '\n';
	};
	TfLiteLogErrorCallback tflite_log_error_callback = [](std::string msg) {
		std::cerr << "[TfLite Error] " << msg << '\n';
	};

	YoloModel yolo_instance;

	auto labels_result =
		read_coco_labels_file("../../EyeAIApp/app/src/main/assets/coco.names");
	EXPECT_RESULT_HAS_VALUE(labels_result);
	const auto& labels = labels_result.value();

	auto result = yolo_instance.create(
		std::move(model_data), labels, gpu_delegate_serialization_dir.string(),
		YOLO_MODEL_TOKEN, tflite_log_warning_callback, tflite_log_error_callback
	);

	EXPECT_RESULT_HAS_VALUE(result);

	std::cout << "Runtime erstellt!\n";

	auto input_image_result =
		load_image_file("../tests/cat.jpg", INPUT_WIDTH, INPUT_HEIGHT);
	EXPECT_RESULT_HAS_VALUE(input_image_result);
	auto& input_image = *input_image_result;
	auto input_image_tensor = image_rgb_255_operator(input_image);

	const auto run_result = yolo_instance.run(input_image_tensor);

	EXPECT_RESULT_HAS_VALUE(run_result);

	const auto& boxes = *run_result;

	EXPECT_EQ(boxes.size(), 1);
	const auto contains_label = [&boxes](std::string_view label) {
		return std::ranges::any_of(
			boxes, [label](const YoloModel::BoundingBox& box) {
				return box.cls_name == label;
			}
		);
	};
	EXPECT_TRUE(contains_label("cat"));
}
