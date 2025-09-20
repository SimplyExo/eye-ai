#pragma once

#include "EyeAICore/TensorBuffer.hpp"
#include "EyeAICore/tflite/TfLiteRuntime.hpp"

class YoloModel {
  public:
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

	YoloModel() = default;

	// Erstellt das Modell
	tl::expected<bool, std::string> create(
		std::vector<int8_t>&& model_data,
		std::vector<std::string> labels,
		std::string_view gpu_delegate_serialization_dir,
		std::string_view model_token,
		TfLiteLogWarningCallback log_warning_callback,
		TfLiteLogErrorCallback log_error_callback,
		bool enable_npu,
		std::string skel_directory_dir
	);

	tl::expected<std::vector<BoundingBox>, std::string>
	run(FloatTensorBuffer<FloatTensorFormat::ImageRGB255>& input);

	[[nodiscard]] std::vector<BoundingBox>
	best_box(std::span<const float> array) const;

	std::span<const int> get_input_shape();

	std::span<const int> get_output_shape();

	size_t num_channel = 0;
	size_t num_elements = 0;

  private:
	[[nodiscard]] std::optional<BoundingBox>
	parse_box(std::span<const float> array, size_t box_index) const;
	[[nodiscard]] std::vector<BoundingBox>
	apply_nms(const std::vector<BoundingBox>& boxes) const;
	static float
	calculate_iou(const BoundingBox& box1, const BoundingBox& box2);

	std::unique_ptr<TfLiteRuntime> runtime;

	std::vector<std::string> labels;

	const float CONFIDENCE_THRESHOLD = 0.5F;
	const float IOU_THRESHOLD = 0.5F;
};
