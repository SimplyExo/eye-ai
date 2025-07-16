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

/// Helper class that wraps the tflite c api
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

	TfLiteReporterUserData reporter_user_data;

	std::vector<std::unique_ptr<Operator>> input_operators;
	std::vector<std::unique_ptr<Operator>> output_operators;

  public:
	using CreateResult =
		tl::expected<std::unique_ptr<TfLiteRuntime>, TfLiteCreateRuntimeError>;

	/**
	 * Create a TfLiteRuntime instance, see TfLiteRuntimeBuilder for a builder
	 * pattern.
	 */
	[[nodiscard]] static CreateResult create(
		std::vector<int8_t>&& model_data,
		std::string_view gpu_delegate_serialization_dir,
		std::string_view model_token,
		std::vector<std::unique_ptr<Operator>>&& input_operators,
		std::vector<std::unique_ptr<Operator>>&& output_operators,
		TfLiteLogWarningCallback log_warning_callback,
		TfLiteLogErrorCallback log_error_callback
	);

	~TfLiteRuntime();

	/**
	 * @brief Run inference on the model, make sure input and output have the
	 * right amount of elements.
	 * @param input input will be modified by input operators!
	 * @param output output will be modified by output operators!
	 */
	[[nodiscard]] std::optional<TfLiteRunInferenceError>
	run_inference(std::span<float> input, std::span<float> output);

	[[nodiscard]] std::span<const int> get_input_shape() const;

	[[nodiscard]] std::span<const int> get_output_shape() const;

	TfLiteRuntime(TfLiteRuntime&&) = delete;
	TfLiteRuntime(const TfLiteRuntime&) = delete;
	void operator=(TfLiteRuntime&&) = delete;
	void operator=(const TfLiteRuntime&) = delete;

  private:
	explicit TfLiteRuntime(
		std::vector<int8_t>&& model_data,
		std::vector<std::unique_ptr<Operator>>&& input_operators,
		std::vector<std::unique_ptr<Operator>>&& output_operators,
		TfLiteReporterUserData error_reporter_user_data
	)
		: model_data(std::move(model_data)),
		  reporter_user_data(error_reporter_user_data),
		  input_operators(std::move(input_operators)),
		  output_operators(std::move(output_operators)) {}

	[[nodiscard]] std::optional<TfLiteInvokeInterpreterError> invoke();

	[[nodiscard]] std::optional<TfLiteLoadInputError>
	load_input(std::span<const float> input);

	[[nodiscard]] std::optional<TfLiteReadOutputError>
	read_output(std::span<float> output);
};

/// Helper class to reduce boilerplate code when creating a TfLiteRuntime
class TfLiteRuntimeBuilder {
  public:
	using Result =
		tl::expected<std::unique_ptr<TfLiteRuntime>, TfLiteCreateRuntimeError>;

	explicit TfLiteRuntimeBuilder(
		std::vector<int8_t>&& model_data,
		std::string_view gpu_delegate_serialization_dir,
		std::string_view model_token,
		TfLiteLogWarningCallback log_warning_callback,
		TfLiteLogErrorCallback log_error_callback
	);

	TfLiteRuntimeBuilder&
	add_input_operator(std::unique_ptr<Operator>&& input_operator);

	TfLiteRuntimeBuilder&
	add_output_operator(std::unique_ptr<Operator>&& output_operator);

	/// all modified configurations of `this` will be discarded after this
	/// method
	[[nodiscard]]
	Result build();

  private:
	std::vector<int8_t> model_data;
	std::string_view gpu_delegate_serialization_dir;
	std::string_view model_token;
	std::vector<std::unique_ptr<Operator>> input_operators;
	std::vector<std::unique_ptr<Operator>> output_operators;
	TfLiteLogWarningCallback log_warning_callback;
	TfLiteLogErrorCallback log_error_callback;
};