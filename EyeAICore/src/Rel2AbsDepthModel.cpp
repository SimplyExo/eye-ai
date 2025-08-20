#include "EyeAICore/Rel2AbsDepthModel.hpp"
#include "EyeAICore/TensorBuffer.hpp"
#include "EyeAICore/utils/Profiling.hpp"

Rel2AbsDepthModel::CreateResult Rel2AbsDepthModel::create(
	std::vector<int8_t>&& model_data,
	std::string_view gpu_delegate_serialization_dir,
	std::string_view model_token,
	TfLiteLogWarningCallback log_warning_callback,
	TfLiteLogErrorCallback log_error_callback
) {
	PROFILE_DEPTH_FUNCTION()

	auto runtime = TfLiteRuntime::create(
		std::move(model_data), gpu_delegate_serialization_dir, model_token,
		FloatTensorFormat::Rel2AbsDepthInput,
		FloatTensorFormat::Rel2AbsDepthCoefficientOutput, log_warning_callback,
		log_error_callback, get_depth_profiling_frame()
	);

	if (!runtime)
		return tl::unexpected(runtime.error());

	return std::make_unique<Rel2AbsDepthModel>(std::move(*runtime));
}

Rel2AbsDepthModel::RunResult Rel2AbsDepthModel::run(
	FloatTensorBuffer<FloatTensorFormat::Rel2AbsDepthInput>& input
) {
	PROFILE_DEPTH_FUNCTION()

	return runtime->run_inference<
		FloatTensorFormat::Rel2AbsDepthInput,
		FloatTensorFormat::Rel2AbsDepthCoefficientOutput>(input);
}