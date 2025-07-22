#include "EyeAICore/YoloModel.hpp"
#include "utils.hpp"
#include <array>
#include <filesystem>
#include <iostream>
#include <ostream>
#include <span>
#include <string>

constexpr const char* GPU_DELEGATE_SERIALIZATION_DIR =
	"/tmp/EyeAICore/gpu_delegate_cache";
constexpr const char* MIDAS_MODEL_TOKEN = "ijustmadethistokenup";
constexpr size_t INPUT_WIDTH = 640;
constexpr size_t INPUT_HEIGHT = 640;
constexpr size_t OUTPUT_BUFFER_SIZE = 84 * 8400;

constexpr static std::array<std::string, 84> COCO_LABELS = {
	"person",		 "bicycle",		 "car",
	"motorcycle",	 "airplane",	 "bus",
	"train",		 "truck",		 "boat",
	"traffic light", "fire hydrant", "stop sign",
	"parking meter", "bench",		 "bird",
	"cat",			 "dog",			 "horse",
	"sheep",		 "cow",			 "elephant",
	"bear",			 "zebra",		 "giraffe",
	"backpack",		 "umbrella",	 "handbag",
	"tie",			 "suitcase",	 "frisbee",
	"skis",			 "snowboard",	 "sports ball",
	"kite",			 "baseball bat", "baseball glove",
	"skateboard",	 "surfboard",	 "tennis racket",
	"bottle",		 "wine glass",	 "cup",
	"fork",			 "knife",		 "spoon",
	"bowl",			 "banana",		 "apple",
	"sandwich",		 "orange",		 "broccoli",
	"carrot",		 "hot dog",		 "pizza",
	"donut",		 "cake",		 "chair",
	"couch",		 "potted plant", "bed",
	"dining table",	 "toilet",		 "tv",
	"laptop",		 "mouse",		 "remote",
	"keyboard",		 "cell phone",	 "microwave",
	"oven",			 "toaster",		 "sink",
	"refrigerator",	 "book",		 "clock",
	"vase",			 "scissors",	 "teddy bear",
	"hair drier",	 "toothbrush"
};

TEST(ObjectDetection, CorrectOutput) {
	std::filesystem::create_directories(GPU_DELEGATE_SERIALIZATION_DIR);

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

	std::vector<std::string> labels(COCO_LABELS.begin(), COCO_LABELS.end());

	auto result = yolo_instance.create(
		std::move(model_data), labels, GPU_DELEGATE_SERIALIZATION_DIR,
		MIDAS_MODEL_TOKEN, tflite_log_warning_callback,
		tflite_log_error_callback
	);

	EXPECT_RESULT_HAS_VALUE(result);

	std::cout << "Runtime erstellt!\n";

	std::vector<float> object_recognition_output(OUTPUT_BUFFER_SIZE);
	auto input_image_result =
		load_image_file("../tests/cat.jpg", INPUT_WIDTH, INPUT_HEIGHT);
	EXPECT_RESULT_HAS_VALUE(input_image_result);
	auto& input_image = *input_image_result;
	for (float& value : input_image) {
		value = std::clamp(value * 255.f, 0.f, 255.f);
	}

	const auto run_result =
		yolo_instance.run(input_image, object_recognition_output);

	EXPECT_RESULT_HAS_VALUE(run_result);

	const auto boxes = yolo_instance.best_box(object_recognition_output);

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
