#include "EyeAICore/MetricDepthModel.hpp"
#include "EyeAICore/DepthModel.hpp"
#include "EyeAICore/Operators.hpp"
#include "EyeAICore/Rel2AbsDepthModel.hpp"
#include "EyeAICore/utils/Profiling.hpp"

MetricDepthModel::MetricDepthModel(
	std::unique_ptr<DepthModel>&& depth_model,
	std::unique_ptr<Rel2AbsDepthModel>&& rel2abs_depth_model
)
	: depth_model(std::move(depth_model)),
	  rel2abs_depth_model(std::move(rel2abs_depth_model)) {}

MetricDepthModel::CreateResult MetricDepthModel::create(
	std::vector<int8_t>&& depth_model_data,
	std::vector<int8_t>&& rel2abs_depth_model_data,
	std::string_view gpu_delegate_serialization_dir,
	std::string_view depth_model_token,
	std::string_view rel2abs_depth_model_token,
	TfLiteLogWarningCallback log_warning_callback,
	TfLiteLogErrorCallback log_error_callback,
	bool enable_npu
) {
	PROFILE_DEPTH_FUNCTION()

	auto depth_model_result = DepthModel::create(
		std::move(depth_model_data), gpu_delegate_serialization_dir,
		depth_model_token, log_warning_callback, log_error_callback, enable_npu
	);
	if (!depth_model_result) {
		return tl::unexpected(depth_model_result.error());
	}

	auto rel2abs_depth_model_result = Rel2AbsDepthModel::create(
		std::move(rel2abs_depth_model_data), gpu_delegate_serialization_dir,
		rel2abs_depth_model_token, log_warning_callback, log_error_callback,
		enable_npu
	);
	if (!rel2abs_depth_model_result) {
		return tl::unexpected(rel2abs_depth_model_result.error());
	}

	return std::make_unique<MetricDepthModel>(
		std::move(depth_model_result.value()),
		std::move(rel2abs_depth_model_result.value())
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

	FloatTensorBuffer<FloatTensorFormat::Rel2AbsDepthInput>
		rel2abs_input_tensor = rel2abs_input_operator(input, relative_depth);

	auto rel2abs_coeffs_result = rel2abs_depth_model->run(rel2abs_input_tensor);
	if (!rel2abs_coeffs_result) {
		return tl::unexpected(rel2abs_coeffs_result.error());
	}

	return rel2abs_operator(relative_depth, *rel2abs_coeffs_result);
}

std::span<const int> MetricDepthModel::get_input_shape() const {
	return depth_model->get_input_shape();
}

std::span<const int> MetricDepthModel::get_output_shape() const {
	return depth_model->get_output_shape();
}