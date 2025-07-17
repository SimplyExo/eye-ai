#include "EyeAICore/YoloModel.hpp"
#include "EyeAICore/Operators.hpp"
#include "EyeAICore/tflite/TfLiteRuntime.hpp"
#include "tl/expected.hpp"
#include <string>
#include <utility>

YoloModel::YoloModel() {}

tl::expected<bool, std::string> YoloModel::create(
	std::vector<int8_t>&& model_data,
	std::vector<std::string> labels,
	std::string_view gpu_delegate_serialization_dir,
	std::string_view model_token,
	TfLiteLogWarningCallback log_warning_callback,
	TfLiteLogErrorCallback log_error_callback
) {
	// Labels laden
	this->labels = std::move(labels);

	std::vector<std::unique_ptr<Operator>> input_operators;
	input_operators.emplace_back(
		std::make_unique<RgbNormalizeOperator>(
			std::array<float, 3>{0.f, 0.f, 0.f},
			std::array<float, 3>{255.f, 255.f, 255.f}
		)
	);

	auto new_runtime =
		TfLiteRuntimeBuilder(
			std::move(model_data), gpu_delegate_serialization_dir, model_token,
			log_warning_callback, log_error_callback
		)
			.add_input_operator(std::make_unique<RgbNormalizeOperator>())
			.build();

	// bei Fehler gebe string aus
	if (!new_runtime.has_value())
		return tl::unexpected("Failed to create YoloModel");

	// wenn keine Fehler auftreten dann bool
	runtime = std::move(new_runtime.value());

	return true;
}

tl::expected<void, std::string>
YoloModel::run(std::span<float> input, std::span<float> output) {
	auto result = runtime->run_inference(input, output);

	if (result.has_value()) {
		return tl::make_unexpected("Inference failed: ");
	}

	return {};
}

std::vector<YoloModel::BoundingBox>
YoloModel::bestBox(std::span<float> array, int numElements, int numChannel) {
	std::vector<BoundingBox> boundingBoxes;

	const size_t totalSize = array.size();
	if (totalSize < static_cast<size_t>(numElements * numChannel)) {
		return {}; // Fehler: zu wenig Daten
	}

	for (int c = 0; c < numElements; ++c) {
		float maxConf = -1.0f;
		int maxIdx = -1;
		int j = 4;
		size_t arrayIdx = c + numElements * j;

		while (j < numChannel) {
			if (arrayIdx >= totalSize)
				break; // Schutz gegen Überlauf

			if (array[arrayIdx] > maxConf) {
				maxConf = array[arrayIdx];
				maxIdx = j - 4;
			}

			++j;
			arrayIdx += numElements;
		}

		if (maxConf > CONFIDENCE_THRESHOLD) {
			// Index prüfen!
			if (maxIdx < 0 || maxIdx >= static_cast<int>(labels.size()))
				continue;

			float cx = array[c];			   // 0
			float cy = array[c + numElements]; // 1
			float w = array[c + numElements * 2];
			float h = array[c + numElements * 3];

			float x1 = cx - w / 2.0f;
			float y1 = cy - h / 2.0f;
			float x2 = cx + w / 2.0f;
			float y2 = cy + h / 2.0f;

			// Bounds check wie in Kotlin
			if (x1 < 0.0f || x1 > 1.0f)
				continue;
			if (y1 < 0.0f || y1 > 1.0f)
				continue;
			if (x2 < 0.0f || x2 > 1.0f)
				continue;
			if (y2 < 0.0f || y2 > 1.0f)
				continue;

			BoundingBox box;
			box.clsName = labels[maxIdx];
			box.cls = maxIdx;
			box.cnf = maxConf;
			box.cx = cx;
			box.cy = cy;
			box.w = w;
			box.h = h;
			box.x1 = x1;
			box.y1 = y1;
			box.x2 = x2;
			box.y2 = y2;

			boundingBoxes.push_back(box);
		}
	}

	if (boundingBoxes.empty()) {
		return {};
	}

	// Non-Maximum Suppression anwenden
	return applyNMS(boundingBoxes);
}

float YoloModel::calculateIoU(
	const YoloModel::BoundingBox& box1,
	const YoloModel::BoundingBox& box2
) {
	float x1 = std::max(box1.x1, box2.x1);
	float y1 = std::max(box1.y1, box2.y1);
	float x2 = std::min(box1.x2, box2.x2);
	float y2 = std::min(box1.y2, box2.y2);

	float intersectionWidth = std::max(0.0f, x2 - x1);
	float intersectionHeight = std::max(0.0f, y2 - y1);
	float intersectionArea = intersectionWidth * intersectionHeight;

	float box1Area = box1.w * box1.h;
	float box2Area = box2.w * box2.h;

	return intersectionArea / (box1Area + box2Area - intersectionArea);
}

std::vector<YoloModel::BoundingBox>
YoloModel::applyNMS(std::vector<YoloModel::BoundingBox>& boxes) {
	// 1. Sortiere nach cnf absteigend
	std::vector<BoundingBox> sortedBoxes = boxes;
	std::sort(
		sortedBoxes.begin(), sortedBoxes.end(),
		[](const BoundingBox& a, const BoundingBox& b) { return a.cnf > b.cnf; }
	);

	std::vector<BoundingBox> selectedBoxes;

	while (!sortedBoxes.empty()) {
		BoundingBox first = sortedBoxes.front();
		selectedBoxes.push_back(first);
		sortedBoxes.erase(sortedBoxes.begin()); // entferne das erste Element

		auto it = sortedBoxes.begin();
		while (it != sortedBoxes.end()) {
			float iou = calculateIoU(first, *it);
			if (iou >= IOU_THRESHOLD) {
				it = sortedBoxes.erase(it); // entferne überschneidende Box
			} else {
				++it;
			}
		}
	}

	return selectedBoxes;
}