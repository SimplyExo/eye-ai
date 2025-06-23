#include "EyeAICore/DepthModel.hpp"
#include "EyeAICore/Operators.hpp"
#include "EyeAICore/tflite/TfLiteRuntime.hpp"

tl::expected<std::unique_ptr<DepthModel>, std::string> DepthModel::create(
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

	auto runtime = TfLiteRuntime::create(
		std::move(model_data), gpu_delegate_serialization_dir, model_token,
		std::move(input_operators), std::move(output_operators),
		log_warning_callback, log_error_callback
	);
	if (!runtime.has_value())
		return tl::unexpected(runtime.error());

	return std::make_unique<DepthModel>(std::move(runtime.value()));
}

tl::expected<void, std::string>
DepthModel::run(std::span<float> input, std::span<float> output) {
	return runtime->run_inference(input, output);
}