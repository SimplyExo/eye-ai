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
	TfLiteLogErrorCallback log_error_callback
) {
	PROFILE_DEPTH_FUNCTION()

	auto depth_model_result = DepthModel::create(
		std::move(depth_model_data), gpu_delegate_serialization_dir,
		depth_model_token, log_warning_callback, log_error_callback
	);
	if (!depth_model_result) {
		return tl::unexpected(depth_model_result.error());
	}

	auto rel2abs_depth_model_result = Rel2AbsDepthModel::create(
		std::move(rel2abs_depth_model_data), gpu_delegate_serialization_dir,
		rel2abs_depth_model_token, log_warning_callback, log_error_callback
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

	auto input_values = input.data();

	const size_t input_pixel_count = input_values.size() / 3;

	std::vector<float> rel2abs_input(input_pixel_count * 4);

	// copy image channels ([0.f, 255.f] to [-1.f, 1.f])
	{
		PROFILE_DEPTH_SCOPE("Loading Rel2Abs input image")

		for (size_t i = 0; i < input_pixel_count; ++i) {
			rel2abs_input[(i * 4) + 0] =
				(input_values[(i * 3) + 0] / 127.5f) - 1.f;
			rel2abs_input[(i * 4) + 1] =
				(input_values[(i * 3) + 1] / 127.5f) - 1.f;
			rel2abs_input[(i * 4) + 2] =
				(input_values[(i * 3) + 2] / 127.5f) - 1.f;
		}
	}

	auto relative_depth_result = depth_model->run_raw(input);
	if (!relative_depth_result)
		return tl::unexpected(relative_depth_result.error());
	auto& relative_depth = *relative_depth_result;

	// copy raw relative depth into 4th channel ([0.f, 1500.f] to [-1.f, 1.f])
	{
		PROFILE_DEPTH_SCOPE("Load raw relative depth into Rel2Abs")

		auto relative_depth_values = relative_depth.data();

		for (size_t i = 0; i < input_pixel_count; ++i) {
			const float remapped_raw_rel_depth =
				(relative_depth_values[i] / 750.f) - 1.f;
			rel2abs_input[(i * 4) + 3] =
				std::clamp(remapped_raw_rel_depth, -1.f, 1.f);
		}
	}

	FloatTensorBuffer<FloatTensorFormat::Rel2AbsDepthInput>
		rel2abs_input_tensor{std::move(rel2abs_input)};

	auto rel2abs_coeffs_result = rel2abs_depth_model->run(rel2abs_input_tensor);
	if (!rel2abs_coeffs_result) {
		return tl::unexpected(rel2abs_coeffs_result.error());
	}

	return rel2abs_operator(relative_depth, *rel2abs_coeffs_result);
}