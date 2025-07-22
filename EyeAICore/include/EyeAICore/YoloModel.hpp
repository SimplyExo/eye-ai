#pragma once

#include "EyeAICore/tflite/TfLiteRuntime.hpp"

class YoloModel {
  public:
	YoloModel();

	struct BoundingBox {
		std::string cls_name;
		float cx = 0; // 0
		float cy = 0; // 1
		float w = 0;
		float h = 0;
		float x1 = 0;
		float y1 = 0;
		float x2 = 0;
		float y2 = 0;
		int cls = 0;
		float cnf = 0;

		bool operator==(const BoundingBox&) const = default;
	};

	// Erstellt das Modell
	tl::expected<bool, std::string> create(
		std::vector<int8_t>&& model_data,
		std::vector<std::string> labels,
		std::string_view gpu_delegate_serialization_dir,
		std::string_view model_token,
		TfLiteLogWarningCallback log_warning_callback,
		TfLiteLogErrorCallback log_error_callback
	);

	tl::expected<void, std::string>
	run(std::span<float> input, std::span<float> output);

	std::vector<YoloModel::BoundingBox> best_box(std::span<float> array);

	std::span<const int> get_input_shape();

	std::span<const int> get_output_shape();

	int num_channel;
	int num_elements;

  private:
	std::unique_ptr<TfLiteRuntime> runtime;

	std::vector<std::string> labels;

	std::vector<YoloModel::BoundingBox>
	apply_nms(std::vector<YoloModel::BoundingBox>& boxes) const;
	static float calculate_iou(
		const YoloModel::BoundingBox& box1,
		const YoloModel::BoundingBox& box2
	);

	const float CONFIDENCE_THRESHOLD = 0.5F;
	const float IOU_THRESHOLD = 0.5F;
};
