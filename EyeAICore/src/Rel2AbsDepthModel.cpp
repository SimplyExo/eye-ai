#include "EyeAICore/Rel2AbsDepthModel.hpp"
#include "EyeAICore/TensorBuffer.hpp"
#include "EyeAICore/utils/Profiling.hpp"

Rel2AbsDepthModel::CreateResult Rel2AbsDepthModel::create(
	std::vector<int8_t>&& model_data,
	std::string_view gpu_delegate_serialization_dir,
	std::string_view model_token,
	TfLiteLogWarningCallback log_warning_callback,
	TfLiteLogErrorCallback log_error_callback,
	std::string npu_skel_directory
) {
	PROFILE_DEPTH_FUNCTION()

	auto runtime = TfLiteRuntime::create(
		std::move(model_data), gpu_delegate_serialization_dir, model_token,
		FloatTensorFormat::Rel2AbsDepthInput,
		FloatTensorFormat::Rel2AbsDepthCoefficientOutput, log_warning_callback,
		log_error_callback, get_depth_profiling_frame(),
		NpuConfiguration::rel2abs, std::move(npu_skel_directory)
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