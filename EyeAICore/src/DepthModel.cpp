#include "EyeAICore/DepthModel.hpp"
#include "EyeAICore/Operators.hpp"
#include "EyeAICore/tflite/TfLiteRuntime.hpp"

tl::expected<std::unique_ptr<DepthModel>, TfLiteCreateRuntimeError>
DepthModel::create(
	std::vector<int8_t>&& model_data,
	std::string_view gpu_delegate_serialization_dir,
	std::string_view model_token,
	TfLiteLogWarningCallback log_warning_callback,
	TfLiteLogErrorCallback log_error_callback
) {
	auto runtime_result = TfLiteRuntime::create(
		std::move(model_data), gpu_delegate_serialization_dir, model_token,
		FloatTensorFormat::MiDaSImageRGBFloat,
		FloatTensorFormat::RawRelativeDepth,
		OperatorChain{MiDaSImageOperator{}},
		OperatorChain{RelativeDepthPostOperator{}}, log_warning_callback,
		log_error_callback
	);
	if (!runtime_result.has_value())
		return tl::unexpected(runtime_result.error());

	return std::make_unique<DepthModel>(std::move(runtime_result.value()));
}

tl::expected<std::unique_ptr<DepthModel>, TfLiteCreateRuntimeError>
DepthModel::create_with_raw_output(
	std::vector<int8_t>&& model_data,
	std::string_view gpu_delegate_serialization_dir,
	std::string_view model_token,
	TfLiteLogWarningCallback log_warning_callback,
	TfLiteLogErrorCallback log_error_callback
) {
	auto runtime_result = TfLiteRuntime::create(
		std::move(model_data), gpu_delegate_serialization_dir, model_token,
		FloatTensorFormat::MiDaSImageRGBFloat,
		FloatTensorFormat::RawRelativeDepth,
		OperatorChain{MiDaSImageOperator{}}, OperatorChain{},
		log_warning_callback, log_error_callback
	);
	if (!runtime_result.has_value())
		return tl::unexpected(runtime_result.error());

	return std::make_unique<DepthModel>(std::move(runtime_result.value()));
}

std::optional<TfLiteRunInferenceError>
DepthModel::run(std::span<float> input, std::span<float> output) {
	return runtime->run_inference(
		input, FloatTensorFormat::ImageRGB255Float, output,
		FloatTensorFormat::RelativeDepth
	);
}

std::optional<TfLiteRunInferenceError>
DepthModel::run_raw(std::span<float> input, std::span<float> output) {
	return runtime->run_inference(
		input, FloatTensorFormat::ImageRGB255Float, output,
		FloatTensorFormat::RawRelativeDepth
	);
}

std::span<const int> DepthModel::get_input_shape() const {
	return runtime->get_input_shape();
}
std::span<const int> DepthModel::get_output_shape() const {
	return runtime->get_output_shape();
}