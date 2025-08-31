#pragma once

#include "EyeAICore/TensorBuffer.hpp"
#include "EyeAICore/tflite/TfLiteRuntime.hpp"
#include "EyeAICore/tflite/TfLiteUtils.hpp"
#include <memory>

class Rel2AbsDepthModel {
  public:
	constexpr static size_t POLYNOMIAL_DEGREE = 4;
	constexpr static size_t COEFFS_COUNT = POLYNOMIAL_DEGREE + 1;

	using CreateResult = tl::
		expected<std::unique_ptr<Rel2AbsDepthModel>, TfLiteCreateRuntimeError>;

	[[nodiscard]] static CreateResult create(
		std::vector<int8_t>&& model_data,
		std::string_view gpu_delegate_serialization_dir,
		std::string_view model_token,
		TfLiteLogWarningCallback log_warning_callback,
		TfLiteLogErrorCallback log_error_callback
	);

	using RunResult = tl::expected<
		FloatTensorBuffer<FloatTensorFormat::Rel2AbsDepthCoefficientOutput>,
		TfLiteRunInferenceError>;

	[[nodiscard]] RunResult
	run(FloatTensorBuffer<FloatTensorFormat::Rel2AbsDepthInput>& input);

	/// @see MetricDepthModel::create
	Rel2AbsDepthModel(std::unique_ptr<TfLiteRuntime>&& runtime)
		: runtime(std::move(runtime)) {}

  private:
	std::unique_ptr<TfLiteRuntime> runtime;
};

FloatTensorBuffer<FloatTensorFormat::Rel2AbsDepthInput> rel2abs_input_operator(
	const FloatTensorBuffer<FloatTensorFormat::ImageRGB255>& rgb,
	const FloatTensorBuffer<FloatTensorFormat::RawRelativeDepth>& depth
);