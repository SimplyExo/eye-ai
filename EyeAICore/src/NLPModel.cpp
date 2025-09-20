#include "EyeAICore/NLPModel.hpp"
#include "EyeAICore/Operators.hpp"
#include "EyeAICore/tflite/TfLiteRuntime.hpp"
#include "EyeAICore/utils/Profiling.hpp"
#include "tl/expected.hpp"
#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>


tl::expected<bool, std::string> NLPModel::create(
	std::vector<int8_t>&& model_data,
	std::string_view gpu_delegate_serialization_dir,
	std::string_view model_token,
	TfLiteLogWarningCallback log_warning_callback,
	TfLiteLogErrorCallback log_error_callback,
	bool enable_npu,
	std::string npu_skel_directory
) {
	PROFILE_OBJECT_FUNCTION()

	auto new_runtime = TfLiteRuntime::create(
		std::move(model_data), gpu_delegate_serialization_dir, model_token,
		FloatTensorFormat::NLPInput, FloatTensorFormat::NLPOutput,
		log_warning_callback, log_error_callback, get_object_profiling_frame(),
		NpuConfiguration::Yolo, enable_npu, std::move(npu_skel_directory)
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

std::span<const int> NLPModel::get_input_shape() {
	return runtime->get_input_shape();
}

std::span<const int> NLPModel::get_output_shape() {
	return runtime->get_output_shape();
}

tl::expected<std::vector<float>, std::string>
NLPModel::run(FloatTensorBuffer<FloatTensorFormat::NLPInput>& input) {
	auto result = runtime->run_inference<
		FloatTensorFormat::NLPInput, FloatTensorFormat::NLPOutput>(
		input
	);

	if (!result) {
		return tl::unexpected(result.error().to_string());
	}

	return to_vector(result->data());
}

std::vector<float> NLPModel::to_vector(std::span<float> in) {
	std::vector<float> out = {};

	for (float f : in) {
		out.push_back(f);
	}

	return out;
}

