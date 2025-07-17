#pragma once

#include "EyeAICore/Operators.hpp"
#include "EyeAICore/tflite/TfLiteRuntime.hpp"
#include "jni.h"

class YoloModel
{
    public:
    YoloModel();

	struct BoundingBox
	{
		std::string clsName = "";
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

	std::vector<YoloModel::BoundingBox> bestBox(std::span<float> array, int numElements, int numChannel);

  	static jobjectArray convertToJavaBoundingBoxArray(JNIEnv* env, const std::vector<YoloModel::BoundingBox>& boxes);

    private:
	  std::unique_ptr<TfLiteRuntime> runtime;

	  std::vector<std::string> labels;

	  std::vector<YoloModel::BoundingBox> applyNMS(std::vector<YoloModel::BoundingBox>& boxes);
	  float calculateIoU(const YoloModel::BoundingBox& box1, const YoloModel::BoundingBox& box2);

	  float CONFIDENCE_THRESHOLD = 0.5F;
	  float IOU_THRESHOLD = 0.5F;
};
