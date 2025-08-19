#pragma once

#include "EyeAICore/TensorBuffer.hpp"
#include "EyeAICore/tflite/TfLiteRuntime.hpp"

/// A utility class for running depth estimation models like MiDaS.
class DepthModel {
  public:
	[[nodiscard]] static tl::
		expected<std::unique_ptr<DepthModel>, TfLiteCreateRuntimeError>
		create(
			std::vector<int8_t>&& model_data,
			std::string_view gpu_delegate_serialization_dir,
			std::string_view model_token,
			TfLiteLogWarningCallback log_warning_callback,
			TfLiteLogErrorCallback log_error_callback
		);

	/// see @ref DepthModel::create
	DepthModel(std::unique_ptr<TfLiteRuntime>&& runtime)
		: runtime(std::move(runtime)) {}

	DepthModel(DepthModel&&) = default;
	DepthModel& operator=(DepthModel&&) = default;

	DepthModel(const DepthModel&) = delete;
	DepthModel& operator=(const DepthModel&) = delete;

	~DepthModel() = default;

	/**
	 * @param input should have 3 * width * height elements.
	 * @param output should have width * height elements.
	 */
	[[nodiscard]] tl::expected<
		FloatTensorBuffer<FloatTensorFormat::RelativeDepth>,
		TfLiteRunInferenceError>
	run(FloatTensorBuffer<FloatTensorFormat::ImageRGB255>& input);

	[[nodiscard]] std::span<const int> get_input_shape() const;

	[[nodiscard]] std::span<const int> get_output_shape() const;

  private:
	std::unique_ptr<TfLiteRuntime> runtime;
};