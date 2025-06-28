#pragma once

#include "EyeAICore/tflite/TfLiteRuntime.hpp"

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

	[[nodiscard]] static tl::expected<std::unique_ptr<DepthModel>, std::string>
	create_with_raw_output(
		std::vector<int8_t>&& model_data,
		std::string_view gpu_delegate_serialization_dir,
		std::string_view model_token,
		TfLiteLogWarningCallback log_warning_callback,
		TfLiteLogErrorCallback log_error_callback
	);

	DepthModel(std::unique_ptr<TfLiteRuntime>&& runtime)
		: runtime(std::move(runtime)) {}

	[[nodiscard]] std::optional<TfLiteRunInferenceError>
	run(std::span<float> input, std::span<float> output);

  private:
	std::unique_ptr<TfLiteRuntime> runtime;
};