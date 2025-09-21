#include "EyeAICore/MetricDepthModel.hpp"
#include "EyeAICore/DepthModel.hpp"
#include "EyeAICore/Operators.hpp"
#include "EyeAICore/utils/Profiling.hpp"

MetricDepthModel::MetricDepthModel(std::unique_ptr<DepthModel>&& depth_model)
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
		depth_model_token, log_warning_callback, log_error_callback, enable_npu,
		npu_skel_directory
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

template<typename T>
constexpr static T remap(T value, T from_min, T from_max, T to_min, T to_max) {
	return (((value - from_min) * (to_max - to_min)) / (from_max - from_min)) +
		   to_min;
}

FloatTensorBuffer<FloatTensorFormat::Rel2AbsDepthInput> rel2abs_input_operator(
	const FloatTensorBuffer<FloatTensorFormat::ImageRGB255>& rgb,
	const FloatTensorBuffer<FloatTensorFormat::RawRelativeDepth>& depth
) {
	PROFILE_DEPTH_FUNCTION()

	const size_t input_pixel_count = rgb.data().size() / 3;

	std::vector<float> rel2abs_input(input_pixel_count * 4);

	auto rgb_values = rgb.data();
	// copy image channels ([0.f, 255.f] to [-1.f, 1.f])
	for (size_t i = 0; i < input_pixel_count; ++i) {
		rel2abs_input[(i * 4) + 0] =
			remap(rgb_values[(i * 3) + 0], 0.f, 255.f, -1.f, 1.f);
		rel2abs_input[(i * 4) + 1] =
			remap(rgb_values[(i * 3) + 1], 0.f, 255.f, -1.f, 1.f);
		rel2abs_input[(i * 4) + 2] =
			remap(rgb_values[(i * 3) + 2], 0.f, 255.f, -1.f, 1.f);
	}

	auto relative_depth_values = depth.data();

	// remap raw relative depth from [0.f, 1500.f] to [-1.f, 1.f]
	for (size_t i = 0; i < input_pixel_count; ++i) {
		const float remapped_raw_rel_depth =
			remap(relative_depth_values[i], 0.f, 1500.f, -1.f, 1.f);
		rel2abs_input[(i * 4) + 3] =
			std::clamp(remapped_raw_rel_depth, -1.f, 1.f);
	}

	return FloatTensorBuffer<FloatTensorFormat::Rel2AbsDepthInput>(
		std::move(rel2abs_input)
	);
}