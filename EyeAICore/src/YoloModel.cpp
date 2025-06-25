#include "EyeAICore/YoloModel.hpp"
#include "EyeAICore/Operators.hpp"
#include "EyeAICore/tflite/TfLiteRuntime.hpp"
#include "tl/expected.hpp"
#include <string>

YoloModel::YoloModel() {}

tl::expected<bool, std::string> YoloModel::create(
	std::vector<int8_t>&& model_data,
	std::string_view gpu_delegate_serialization_dir,
	std::string_view model_token,
	TfLiteLogWarningCallback log_warning_callback,
	TfLiteLogErrorCallback log_error_callback
) {
	std::vector<std::unique_ptr<Operator>> input_operators;
	input_operators.emplace_back(std::make_unique<RgbNormalizeOperator>());

	std::vector<std::unique_ptr<Operator>> output_operators;
	output_operators.emplace_back(std::make_unique<MinMaxOperator>());

	auto new_runtime = TfLiteRuntime::create(
		std::move(model_data), gpu_delegate_serialization_dir, model_token,
		std::move(input_operators), std::move(output_operators),
		log_warning_callback, log_error_callback
	);

	// bei Fehler gebe string aus
	if (!new_runtime.has_value())
		return tl::unexpected(new_runtime.error());

	// wenn keine Fehler auftreten dann bool
	runtime = std::move(new_runtime.value());

	return true;
}

tl::expected<void, std::string>
YoloModel::run(std::span<float> input, std::span<float> output) {
	return runtime->run_inference(input, output);
}
