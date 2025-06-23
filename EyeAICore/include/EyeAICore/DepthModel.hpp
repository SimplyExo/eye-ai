#pragma once

#include "EyeAICore/tflite/TfLiteRuntime.hpp"

class DepthModel {
  public:
	[[nodiscard]] static tl::expected<std::unique_ptr<DepthModel>, std::string>
	create(
		std::vector<int8_t>&& model_data,
		std::string_view gpu_delegate_serialization_dir,
		std::string_view model_token,
		TfLiteLogWarningCallback log_warning_callback,
		TfLiteLogErrorCallback log_error_callback
	);

	DepthModel(std::unique_ptr<TfLiteRuntime>&& runtime)
		: runtime(std::move(runtime)) {}

	[[nodiscard]] tl::expected<void, std::string>
	run(std::span<float> input, std::span<float> output);

  private:
	std::unique_ptr<TfLiteRuntime> runtime;
};