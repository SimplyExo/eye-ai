#include "EyeAICore/MetricDepthModel.hpp"
#include "EyeAICore/DepthModel.hpp"
#include "EyeAICore/Operators.hpp"
#include "EyeAICore/utils/Profiling.hpp"

MetricDepthModel::MetricDepthModel(
	std::unique_ptr<DepthModel>&& depth_model
)
	: depth_model(std::move(depth_model)) {}

MetricDepthModel::CreateResult MetricDepthModel::create(
	std::vector<int8_t>&& depth_model_data,
	std::string_view gpu_delegate_serialization_dir,
	std::string_view depth_model_token,
	TfLiteLogWarningCallback log_warning_callback,
	TfLiteLogErrorCallback log_error_callback,
	bool enable_npu,
	std::string npu_skel_directory
) {
	PROFILE_DEPTH_FUNCTION()

	auto depth_model_result = DepthModel::create(
		std::move(depth_model_data), gpu_delegate_serialization_dir,
		depth_model_token, log_warning_callback, log_error_callback,
		enable_npu, npu_skel_directory
	);
	if (!depth_model_result) {
		return tl::unexpected(depth_model_result.error());
	}

	return std::make_unique<MetricDepthModel>(
		std::move(depth_model_result.value())
	);
}

MetricDepthModel::RunResult MetricDepthModel::run(
	FloatTensorBuffer<FloatTensorFormat::ImageRGB255>& input
) {
	PROFILE_DEPTH_FUNCTION()

	auto relative_depth_result = depth_model->run_raw(input);
	if (!relative_depth_result)
		return tl::unexpected(relative_depth_result.error());
	auto& relative_depth = *relative_depth_result;

	return rel2abs_operator(relative_depth, REL2ABS_COEFFS);
}

std::span<const int> MetricDepthModel::get_input_shape() const {
	return depth_model->get_input_shape();
}

std::span<const int> MetricDepthModel::get_output_shape() const {
	return depth_model->get_output_shape();
}