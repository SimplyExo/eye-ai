#include "EyeAICore/YoloModel.hpp"
#include "EyeAICore/utils/Errors.hpp"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <ostream>
#include <span>

#include <opencv2/opencv.hpp>
#include <string>

constexpr const char* GPU_DELEGATE_SERIALIZATION_DIR =
	"/tmp/EyeAICore/gpu_delegate_cache";
constexpr const char* MIDAS_MODEL_TOKEN = "ijustmadethistokenup";
constexpr size_t INPUT_WIDTH = 640;
constexpr size_t INPUT_HEIGHT = 640;
constexpr size_t OUTPUT_BUFFER_SIZE = 84 * 8400;
constexpr std::array<float, 3> MEAN = {123.675f, 116.28f, 103.53f};
constexpr std::array<float, 3> STDDEV = {58.395f, 57.12f, 57.375f};

template<typename T>
static tl::expected<std::vector<T>, std::string>
read_binary_file(const std::filesystem::path& filepath);

void print_vector(const std::vector<float>& v) {
	for (float i : v) {
		std::cout << i << std::endl;
	}
}

static std::vector<float> image_to_rgb(const std::string& image_path) {
	cv::Mat img = cv::imread(image_path);
	cv::resize(img, img, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));

	// cv::imshow(image_path, img);
	// cv::waitKey(0);

	std::vector<float> out = {};

	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			// Für Farbbilder (3 Kanäle: BGR)
			cv::Vec3b color = img.at<cv::Vec3b>(y, x);
			float blue = color[0];
			float green = color[1];
			float red = color[2];

			out.push_back(red / 255.0);
			out.push_back(green / 255.0);
			out.push_back(blue / 255.0);
		}
	}

	return out;
}

std::vector<std::vector<float>> reshape(
	std::vector<float> input,
	unsigned int values,
	unsigned int detections
) {
	std::vector<std::vector<float>> output(
		detections, std::vector<float>(values)
	);

	for (int i = 0; i < detections; ++i) {
		for (int j = 0; j < values; ++j) {
			output[i][j] = input[i * values + j];
		}
	}

	return output;
}

const std::vector<std::string> coco_labels = {
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

int main() {
	std::filesystem::create_directories(GPU_DELEGATE_SERIALIZATION_DIR);

	auto model_data_result =
		read_binary_file<int8_t>("/home/thomas/StudioProjects/eye-ai/EyeAIApp/"
								 "app/src/main/assets/yolo11n_float32.tflite");
	if (!model_data_result.has_value()) {
		std::cout << "Failed to read model file: " << model_data_result.error();
		return 1;
	}
	auto& model_data = model_data_result.value();

	TfLiteLogWarningCallback tflite_log_warning_callback = [](std::string msg) {
		std::cout << "[TfLite Warning] " << msg << std::endl;
	};
	TfLiteLogErrorCallback tflite_log_error_callback = [](std::string msg) {
		std::cout << "[TfLite Error] " << msg << std::endl;
	};

	YoloModel yolo_instance;

	auto result = yolo_instance.create(
		std::move(model_data), coco_labels, GPU_DELEGATE_SERIALIZATION_DIR,
		MIDAS_MODEL_TOKEN, tflite_log_warning_callback,
		tflite_log_error_callback
	);

	if (!result.has_value()) {
		std::cout << "Could not create YoloModel: " << result.error();
		return 1;
	}

	std::cout << "Runtime erstellt!\n";

	// Bild laden und auf 640x640 skalieren (für die Visualisierung)
	cv::Mat img = cv::imread("/home/thomas/Downloads/auto.jpg");
	if (img.empty()) {
		std::cerr << "Bild konnte nicht geladen werden!\n";
		return 1;
	}
	cv::resize(img, img, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));

	std::vector<float> object_recognition_output(OUTPUT_BUFFER_SIZE);
	auto input_image = image_to_rgb("/home/thomas/Downloads/auto.jpg");

	const auto exec = yolo_instance.run(input_image, object_recognition_output);

	if (!exec.has_value()) {
		std::cout << "Failed to run calculation: " << exec.error();
		return 1;
	} else {
		std::cout << "Success running calculation!\n";
	}

	auto reshaped_output = reshape(object_recognition_output, 84, 8400);
	std::cout << "Werte Ergebnisse aus..." << std::endl;

	float confidenceThreshold = 0.5f;
	float nmsThreshold = 0.4f;

	// Vektoren für NMS und Zeichnung
	std::vector<cv::Rect> boxes;
	std::vector<float> confidences;
	std::vector<int> classIds;

	for (int i = 0; i < 8400; ++i) {
		float x = reshaped_output[i][0];
		float y = reshaped_output[i][1];
		float w = reshaped_output[i][2];
		float h = reshaped_output[i][3];

		// Klassenscores durchsuchen
		float maxConf = 0;
		int classId = -1;
		for (int j = 4; j < 84; ++j) {
			if (reshaped_output[i][j] > maxConf) {
				maxConf = reshaped_output[i][j];
				classId = j - 4;
			}
		}

		if (maxConf > confidenceThreshold) {
			// Box in Pixelkoordinaten (für 640x640 Bild) umrechnen
			int left = static_cast<int>((x - w / 2.0f) * INPUT_WIDTH);
			int top = static_cast<int>((y - h / 2.0f) * INPUT_HEIGHT);
			int width = static_cast<int>(w * INPUT_WIDTH);
			int height = static_cast<int>(h * INPUT_HEIGHT);

			// Clippen, falls Box über Bildgrenzen hinausgeht
			left = std::max(0, left);
			top = std::max(0, top);
			if (left + width > INPUT_WIDTH)
				width = INPUT_WIDTH - left;
			if (top + height > INPUT_HEIGHT)
				height = INPUT_HEIGHT - top;

			boxes.emplace_back(left, top, width, height);
			confidences.push_back(maxConf);
			classIds.push_back(classId);

			std::cout << "Box: (" << left << "," << top << ")-("
					  << (left + width) << "," << (top + height)
					  << ") Klasse: " << coco_labels[classId]
					  << " Konfidenz: " << maxConf << std::endl;
		}
	}

	// NMS ausführen (benötigt OpenCV >= 4.4)
	std::vector<int> indices;
	cv::dnn::NMSBoxes(
		boxes, confidences, confidenceThreshold, nmsThreshold, indices
	);

	// Bounding Boxes zeichnen
	for (int idx : indices) {
		const cv::Rect& box = boxes[idx];
		cv::rectangle(img, box, cv::Scalar(0, 255, 0), 2);

		std::string label = coco_labels[classIds[idx]] + ": " +
							cv::format("%.2f", confidences[idx]);
		int baseLine;
		cv::Size labelSize =
			cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

		int top = std::max(box.y, labelSize.height);
		cv::rectangle(
			img, cv::Point(box.x, top - labelSize.height),
			cv::Point(box.x + labelSize.width, top + baseLine),
			cv::Scalar(0, 255, 0), cv::FILLED
		);
		cv::putText(
			img, label, cv::Point(box.x, top), cv::FONT_HERSHEY_SIMPLEX, 0.5,
			cv::Scalar(0, 0, 0), 1
		);
	}

	// Ergebnis anzeigen und speichern
	cv::imshow("YOLO11n Detection", img);
	cv::waitKey(0);
	cv::imwrite("yolo11n_detected.jpg", img);

	return 0;
}

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
