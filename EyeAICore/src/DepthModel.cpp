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
	auto runtime =
		TfLiteRuntimeBuilder(
			std::move(model_data), gpu_delegate_serialization_dir, model_token,
			log_warning_callback, log_error_callback
		)
			.add_input_operator(std::make_unique<RgbNormalizeOperator>())
			.add_output_operator(std::make_unique<MinMaxOperator>())
			.build();
	if (!runtime.has_value())
		return tl::unexpected(runtime.error());

	return std::make_unique<DepthModel>(std::move(runtime.value()));
}

tl::expected<std::unique_ptr<DepthModel>, std::string>
DepthModel::create_with_raw_output(
	std::vector<int8_t>&& model_data,
	std::string_view gpu_delegate_serialization_dir,
	std::string_view model_token,
	TfLiteLogWarningCallback log_warning_callback,
	TfLiteLogErrorCallback log_error_callback
) {
	auto runtime =
		TfLiteRuntimeBuilder(
			std::move(model_data), gpu_delegate_serialization_dir, model_token,
			log_warning_callback, log_error_callback
		)
			.add_input_operator(std::make_unique<RgbNormalizeOperator>())
			.build();
	if (!runtime.has_value())
		return tl::unexpected(runtime.error());

	return std::make_unique<DepthModel>(std::move(runtime.value()));
}

tl::expected<void, std::string>
DepthModel::run(std::span<float> input, std::span<float> output) {
	return runtime->run_inference(input, output);
}