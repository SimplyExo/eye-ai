#pragma once

#include "EyeAICore/Operators.hpp"
#include "TfLiteUtils.hpp"
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

using TfLiteLogWarningCallback = void (*)(std::string);
using TfLiteLogErrorCallback = void (*)(std::string);

/// Callbacks invoked by the tflite runtime, passed around as void* user_data
struct TfLiteReporterUserData {
	TfLiteLogWarningCallback log_warning_callback;
	TfLiteLogErrorCallback log_error_callback;

	explicit TfLiteReporterUserData(
		TfLiteLogWarningCallback log_warning_callback,
		TfLiteLogErrorCallback log_error_callback
	)
		: log_warning_callback(log_warning_callback),
		  log_error_callback(log_error_callback) {}
};

class ProfilingFrame;

/// Helper class that wraps the tflite c api
class TfLiteRuntime {
	std::vector<int8_t> model_data;
	FloatTensorFormat model_input_format;
	FloatTensorFormat model_output_format;
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

	TfLiteReporterUserData reporter_user_data;

	ProfilingFrame& profiling_frame;

	std::vector<std::unique_ptr<OperatorBase>> input_operators;
	std::vector<std::unique_ptr<OperatorBase>> output_operators;

  public:
	using CreateResult =
		tl::expected<std::unique_ptr<TfLiteRuntime>, TfLiteCreateRuntimeError>;

	/**
	 * Create a TfLiteRuntime instance, see TfLiteRuntimeBuilder for a builder
	 * pattern.
	 */
	template<typename... InputOps, typename... OutputOps>
	[[nodiscard]] static CreateResult create(
		std::vector<int8_t>&& model_data,
		std::string_view gpu_delegate_serialization_dir,
		std::string_view model_token,
		FloatTensorFormat model_input_format,
		FloatTensorFormat model_output_format,
		OperatorChain<InputOps...>&& input_operators,
		OperatorChain<OutputOps...>&& output_operators,
		TfLiteLogWarningCallback log_warning_callback,
		TfLiteLogErrorCallback log_error_callback,
		ProfilingFrame& profiling_frame
	) {
		return create_impl(
			std::move(model_data), gpu_delegate_serialization_dir, model_token,
			model_input_format, model_output_format,
			std::move(input_operators).to_runtime_base(),
			std::move(output_operators).to_runtime_base(), log_warning_callback,
			log_error_callback, profiling_frame
		);
	}

	~TfLiteRuntime();

	/**
	 * @brief Run inference on the model, make sure input and output have the
	 * right amount of elements.
	 * @param input input will be modified by input operators!
	 * @param output output will be modified by output operators!
	 */
	[[nodiscard]] std::optional<TfLiteRunInferenceError> run_inference(
		std::span<float> input,
		FloatTensorFormat input_format,
		std::span<float> output,
		FloatTensorFormat expected_output_format
	);

	[[nodiscard]] std::span<const int> get_input_shape() const;

	[[nodiscard]] std::span<const int> get_output_shape() const;

	TfLiteRuntime(TfLiteRuntime&&) = delete;
	TfLiteRuntime(const TfLiteRuntime&) = delete;
	void operator=(TfLiteRuntime&&) = delete;
	void operator=(const TfLiteRuntime&) = delete;

  private:
	[[nodiscard]] static CreateResult create_impl(
		std::vector<int8_t>&& model_data,
		std::string_view gpu_delegate_serialization_dir,
		std::string_view model_token,
		FloatTensorFormat model_input_format,
		FloatTensorFormat model_output_format,
		std::vector<std::unique_ptr<OperatorBase>>&& input_operators,
		std::vector<std::unique_ptr<OperatorBase>>&& output_operators,
		TfLiteLogWarningCallback log_warning_callback,
		TfLiteLogErrorCallback log_error_callback,
		ProfilingFrame& profiling_frame
	);

	explicit TfLiteRuntime(
		std::vector<int8_t>&& model_data,
		FloatTensorFormat model_input_format,
		FloatTensorFormat model_output_format,
		std::vector<std::unique_ptr<OperatorBase>>&& input_operators,
		std::vector<std::unique_ptr<OperatorBase>>&& output_operators,
		TfLiteReporterUserData error_reporter_user_data,
		ProfilingFrame& profiling_frame
	)
		: model_data(std::move(model_data)),
		  model_input_format(model_input_format),
		  model_output_format(model_output_format),
		  reporter_user_data(error_reporter_user_data),
		  profiling_frame(profiling_frame),
		  input_operators(std::move(input_operators)),
		  output_operators(std::move(output_operators)) {}

	[[nodiscard]] std::optional<TfLiteInvokeInterpreterError> invoke();

	[[nodiscard]] std::optional<TfLiteLoadInputError>
	load_input(std::span<const float> input);

	[[nodiscard]] std::optional<TfLiteReadOutputError>
	read_output(std::span<float> output);
};