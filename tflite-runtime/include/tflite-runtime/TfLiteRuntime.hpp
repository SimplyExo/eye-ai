#pragma once

#include "TfLiteUtils.hpp"
#if TFLITE_RUNTIME_USE_PREBUILT_TFLITE
#include "tflite/c/c_api.h" // IWYU pragma: export
#include "tflite/delegates/gpu/delegate.h"
#include <QNN/QnnTFLiteDelegate.h>
#else
#include "tensorflow/lite/c/c_api.h" // IWYU pragma: export
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif
#include "tl/expected.hpp"
#include <memory>
#include <span>
#include <string_view>

using TfLiteLogWarningCallback = void (*)(const char*);
using TfLiteLogErrorCallback = void (*)(const char*);

/// Callbacks invoked by the tflite runtime, passed around as void*
/// user_data
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
	std::span<const int8_t> model_data;
	std::unique_ptr<TfLiteModel, decltype(&TfLiteModelDelete)> model{
		nullptr, TfLiteModelDelete
	};
	std::unique_ptr<TfLiteInterpreter, decltype(&TfLiteInterpreterDelete)>
		interpreter{nullptr, TfLiteInterpreterDelete};
	std::unique_ptr<
		TfLiteInterpreterOptions,
		decltype(&TfLiteInterpreterOptionsDelete)>
		interpreter_options{nullptr, TfLiteInterpreterOptionsDelete};
	/// can be null if GPU delegate are not supported on this device
	std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> gpu_delegate{
		nullptr, TfLiteGpuDelegateV2Delete
	};
	/// can be null if NPU delegate are not supported on this device
	std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)> npu_delegate{
		nullptr, null_delegate_delete
	};

	TfLiteReporterUserData reporter_user_data;

  public:
	using CreateResult = tl::expected<std::unique_ptr<TfLiteRuntime>, ErrorMsg>;

	/// Create a TfLiteRuntime instance
	[[nodiscard]] static CreateResult create(
		std::span<const int8_t> model_data,
		std::string_view delegate_serialization_dir,
		std::string_view model_token,
		TfLiteLogWarningCallback log_warning_callback,
		TfLiteLogErrorCallback log_error_callback,
		NpuConfiguration npu_config,
		bool enable_npu,
		std::string_view skel_library_dir
	);

	~TfLiteRuntime();

	/**
	 * @brief Run inference on the model, make sure input and output have the
	 * right amount of elements.
	 * @param input input will be modified by input operators, should be in
	 * format model_input_format
	 * @param output output will be modified by output operators, should be in
	 * format model_output_format
	 */
	[[nodiscard]] std::optional<ErrorMsg>
	run_inference(std::span<float> input, std::span<float> output);

	[[nodiscard]] std::span<const int> get_input_shape() const;

	[[nodiscard]] std::span<const int> get_output_shape() const;

	TfLiteRuntime(TfLiteRuntime&&) = delete;
	TfLiteRuntime(const TfLiteRuntime&) = delete;
	void operator=(TfLiteRuntime&&) = delete;
	void operator=(const TfLiteRuntime&) = delete;

  private:
	explicit TfLiteRuntime(
		std::span<const int8_t> model_data,
		TfLiteReporterUserData error_reporter_user_data
	)
		: model_data(model_data), reporter_user_data(error_reporter_user_data) {
	}

	[[nodiscard]] std::optional<ErrorMsg> invoke();

	[[nodiscard]] std::optional<ErrorMsg>
	load_input(std::span<const float> input);

	[[nodiscard]] std::optional<ErrorMsg> read_output(std::span<float> output);
};
