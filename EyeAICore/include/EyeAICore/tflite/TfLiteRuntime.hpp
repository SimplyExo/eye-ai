#pragma once

#include "EyeAICore/Operators.hpp"
#if EYE_AI_CORE_USE_PREBUILT_TFLITE
#include "tflite/c/c_api.h" // IWYU pragma: export
#include "tflite/delegates/gpu/delegate.h"
#else
#include "tensorflow/lite/c/c_api.h" // IWYU pragma: export
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <tl/expected.hpp>

using TfLiteLogWarningCallback = void (*)(std::string);
using TfLiteLogErrorCallback = void (*)(std::string);

struct TfLiteErrorReporterUserData {
	TfLiteLogWarningCallback log_warning_callback;
	TfLiteLogErrorCallback log_error_callback;
};

/** Helper class that wraps the tflite c api */
class TfLiteRuntime {
	std::vector<int8_t> model_data;
	std::unique_ptr<TfLiteModel, decltype(&TfLiteModelDelete)> model{
		nullptr, TfLiteModelDelete
	};
	std::unique_ptr<TfLiteInterpreter, decltype(&TfLiteInterpreterDelete)>
		interpreter{nullptr, TfLiteInterpreterDelete};
	std::unique_ptr<
		TfLiteInterpreterOptions,
		decltype(&TfLiteInterpreterOptionsDelete)>
		interpreter_options{nullptr, TfLiteInterpreterOptionsDelete};
	/// can be null if GPU delegates are not supported on this device
	std::unique_ptr<TfLiteDelegate, decltype(&TfLiteGpuDelegateV2Delete)>
		gpu_delegate{nullptr, TfLiteGpuDelegateV2Delete};

	TfLiteErrorReporterUserData error_reporter_user_data;

	std::vector<std::unique_ptr<Operator>> input_operators;
	std::vector<std::unique_ptr<Operator>> output_operators;

  public:
	[[nodiscard]] static tl::
		expected<std::unique_ptr<TfLiteRuntime>, std::string>
		create(
			std::vector<int8_t>&& model_data,
			std::string_view gpu_delegate_serialization_dir,
			std::string_view model_token,
			std::vector<std::unique_ptr<Operator>>&& input_operators,
			std::vector<std::unique_ptr<Operator>>&& output_operators,
			TfLiteLogWarningCallback log_warning_callback,
			TfLiteLogErrorCallback log_error_callback
		);

	~TfLiteRuntime();

	TfLiteRuntime(TfLiteRuntime&&) = delete;
	TfLiteRuntime(const TfLiteRuntime&) = delete;
	void operator=(TfLiteRuntime&&) = delete;
	void operator=(const TfLiteRuntime&) = delete;

	/// input is going to be processed by input operators, so it will be
	/// modified!
	[[nodiscard]] tl::expected<void, std::string>
	run_inference(std::span<float> input, std::span<float> output);

  private:
	explicit TfLiteRuntime(
		std::vector<int8_t>&& model_data,
		std::vector<std::unique_ptr<Operator>>&& input_operators,
		std::vector<std::unique_ptr<Operator>>&& output_operators,
		TfLiteErrorReporterUserData error_reporter_user_data
	)
		: model_data(std::move(model_data)),
		  input_operators(std::move(input_operators)),
		  output_operators(std::move(output_operators)),
		  error_reporter_user_data(error_reporter_user_data) {}

	[[nodiscard]] tl::expected<void, std::string> invoke();

	[[nodiscard]] tl::expected<void, std::string>
	load_input(std::span<const float> input);

	[[nodiscard]] tl::expected<void, std::string>
	read_output(std::span<float> output);
};