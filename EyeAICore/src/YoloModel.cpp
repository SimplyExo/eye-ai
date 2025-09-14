#include "EyeAICore/YoloModel.hpp"
#include "EyeAICore/Operators.hpp"
#include "EyeAICore/tflite/TfLiteRuntime.hpp"
#include "EyeAICore/utils/Profiling.hpp"
#include "tl/expected.hpp"
#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

tl::expected<bool, std::string> YoloModel::create(
	std::vector<int8_t>&& model_data,
	std::vector<std::string> coco_labels,
	std::string_view gpu_delegate_serialization_dir,
	std::string_view model_token,
	TfLiteLogWarningCallback log_warning_callback,
	TfLiteLogErrorCallback log_error_callback,
	bool enable_npu
) {
	PROFILE_OBJECT_FUNCTION()

	// Labels laden
	this->labels = std::move(coco_labels);

	auto new_runtime = TfLiteRuntime::create(
		std::move(model_data), gpu_delegate_serialization_dir, model_token,
		FloatTensorFormat::YoloImageRGB, FloatTensorFormat::YoloOutput,
		log_warning_callback, log_error_callback, get_object_profiling_frame(),
		enable_npu, NpuConfiguration::Yolo
	);

	// bei Fehler gebe string aus
	// TODO: Better Error Message
	if (!new_runtime.has_value())
		return tl::unexpected("Failed to create YoloModel");

	// wenn keine Fehler auftreten dann bool
	runtime = std::move(new_runtime.value());

	num_channel = runtime->get_output_shape()[1];
	num_elements = runtime->get_output_shape()[2];

	return true;
}

std::span<const int> YoloModel::get_input_shape() {
	return runtime->get_input_shape();
}

std::span<const int> YoloModel::get_output_shape() {
	return runtime->get_output_shape();
}

tl::expected<std::vector<YoloModel::BoundingBox>, std::string>
YoloModel::run(FloatTensorBuffer<FloatTensorFormat::ImageRGB255>& input) {
	PROFILE_OBJECT_FUNCTION()

	auto preprocessed_input = yolo_image_operator(input);

	auto result = runtime->run_inference<
		FloatTensorFormat::YoloImageRGB, FloatTensorFormat::YoloOutput>(
		preprocessed_input
	);

	if (!result) {
		return tl::unexpected(result.error().to_string());
	}

	return best_box(result->data());
}

std::vector<YoloModel::BoundingBox>
YoloModel::best_box(std::span<const float> array) const {
	PROFILE_OBJECT_FUNCTION()

	std::vector<BoundingBox> boundingBoxes;
	const size_t actual_size = num_elements * num_channel;

	if (array.size() < actual_size) {
		return {}; // Fehler: zu wenig Daten
	}

	for (size_t c = 0; c < num_elements; ++c) {
		const auto box = parse_box(array, c);
		if (box.has_value()) {
			boundingBoxes.push_back(box.value());
		}
	}

	// Non-Maximum Suppression anwenden
	return apply_nms(boundingBoxes);
}

std::optional<YoloModel::BoundingBox>
YoloModel::parse_box(std::span<const float> array, size_t box_index) const {
	float maxConf = -1.0f;
	int maxIdx = -1;
	size_t j = 4;
	size_t arrayIdx = box_index + (num_elements * j);

	while (j < num_channel) {
		if (arrayIdx >= array.size())
			break; // Schutz gegen Überlauf

		if (array[arrayIdx] > maxConf) {
			maxConf = array[arrayIdx];
			maxIdx = static_cast<int>(j - 4);
		}

		++j;
		arrayIdx += num_elements;
	}

	for (size_t i = 4; i < num_channel; ++i) {
		if (arrayIdx >= array.size())
			break;

		const float conf = array[arrayIdx];
		if (conf > maxConf) {
			maxConf = conf;
			maxIdx = static_cast<int>(i - 4);
		}

		arrayIdx += num_elements;
	}

	if (maxConf < CONFIDENCE_THRESHOLD)
		return std::nullopt;

	// Index prüfen!
	if (maxIdx < 0 || std::cmp_greater_equal(maxIdx, labels.size()))
		return std::nullopt;

	const float cx = array[box_index + (num_elements * 0)];
	const float cy = array[box_index + (num_elements * 1)];
	const float w = array[box_index + (num_elements * 2)];
	const float h = array[box_index + (num_elements * 3)];

	const float x1 = cx - (w / 2.0f);
	const float y1 = cy - (h / 2.0f);
	const float x2 = cx + (w / 2.0f);
	const float y2 = cy + (h / 2.0f);

	// Bounds check wie in Kotlin
	if (x1 < 0.0f || x1 > 1.0f)
		return std::nullopt;
	if (y1 < 0.0f || y1 > 1.0f)
		return std::nullopt;
	if (x2 < 0.0f || x2 > 1.0f)
		return std::nullopt;
	if (y2 < 0.0f || y2 > 1.0f)
		return std::nullopt;

	return BoundingBox{
		.cls_name = labels[maxIdx],
		.cx = cx,
		.cy = cy,
		.w = w,
		.h = h,
		.x1 = x1,
		.y1 = y1,
		.x2 = x2,
		.y2 = y2,
		.cls = maxIdx,
		.cnf = maxConf
	};
}

float YoloModel::calculate_iou(
	const YoloModel::BoundingBox& box1,
	const YoloModel::BoundingBox& box2
) {
	const float x1 = std::max(box1.x1, box2.x1);
	const float y1 = std::max(box1.y1, box2.y1);
	const float x2 = std::min(box1.x2, box2.x2);
	const float y2 = std::min(box1.y2, box2.y2);

	const float intersectionWidth = std::max(0.0f, x2 - x1);
	const float intersectionHeight = std::max(0.0f, y2 - y1);
	const float intersectionArea = intersectionWidth * intersectionHeight;

	const float box1Area = box1.w * box1.h;
	const float box2Area = box2.w * box2.h;

	return intersectionArea / (box1Area + box2Area - intersectionArea);
}

std::vector<YoloModel::BoundingBox>
YoloModel::apply_nms(const std::vector<YoloModel::BoundingBox>& boxes) const {
	PROFILE_OBJECT_FUNCTION()

	if (boxes.empty())
		return boxes;

	// 1. Sortiere nach cnf absteigend
	std::vector<BoundingBox> sortedBoxes = boxes;
	std::ranges::sort(
		sortedBoxes,
		[](const BoundingBox& a, const BoundingBox& b) { return a.cnf > b.cnf; }
	);

	std::vector<BoundingBox> selectedBoxes;

	while (!sortedBoxes.empty()) {
		BoundingBox const first = sortedBoxes.front();
		selectedBoxes.push_back(first);
		sortedBoxes.erase(sortedBoxes.begin()); // entferne das erste Element

		auto it = sortedBoxes.begin();
		while (it != sortedBoxes.end()) {
			const float iou = calculate_iou(first, *it);
			if (iou >= IOU_THRESHOLD) {
				it = sortedBoxes.erase(it); // entferne überschneidende Box
			} else {
				++it;
			}
		}
	}

	return selectedBoxes;
}