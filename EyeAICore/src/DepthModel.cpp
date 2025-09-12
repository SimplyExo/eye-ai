#include "EyeAICore/DepthModel.hpp"
#include "EyeAICore/Operators.hpp"
#include "EyeAICore/tflite/TfLiteRuntime.hpp"
#include "EyeAICore/utils/Profiling.hpp"

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
		FloatTensorFormat::MiDaSImageRGB, FloatTensorFormat::RawRelativeDepth,
		log_warning_callback, log_error_callback, get_depth_profiling_frame(),
		true
	);
	if (!runtime_result.has_value())
		return tl::unexpected(runtime_result.error());

	return std::make_unique<DepthModel>(std::move(runtime_result.value()));
}

DepthModel::RunResult
DepthModel::run(FloatTensorBuffer<FloatTensorFormat::ImageRGB255>& input) {
	auto preprocessed_input = midas_image_operator(input);

	auto run_result = runtime->run_inference<
		FloatTensorFormat::MiDaSImageRGB, FloatTensorFormat::RawRelativeDepth>(
		preprocessed_input
	);
	if (!run_result) {
		return tl::unexpected(run_result.error());
	}

	auto postprocessed_output = raw_relative_depth_post_operator(*run_result);
	return postprocessed_output;
}

DepthModel::RunRawResult
DepthModel::run_raw(FloatTensorBuffer<FloatTensorFormat::ImageRGB255>& input) {
	auto preprocessed_input = midas_image_operator(input);

	auto run_raw_result = runtime->run_inference<
		FloatTensorFormat::MiDaSImageRGB, FloatTensorFormat::RawRelativeDepth>(
		preprocessed_input
	);

	return run_raw_result;
}

std::span<const int> DepthModel::get_input_shape() const {
	return runtime->get_input_shape();
}
std::span<const int> DepthModel::get_output_shape() const {
	return runtime->get_output_shape();
}