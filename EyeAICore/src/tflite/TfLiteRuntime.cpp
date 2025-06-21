#include "EyeAICore/tflite/TfLiteRuntime.hpp"
#include "EyeAICore/tflite/TfLiteUtils.hpp"
#include "EyeAICore/utils/Profiling.hpp"

#include <format>

#if EYE_AI_CORE_USE_PREBUILT_TFLITE
#include <tflite/c/c_api_experimental.h>
#else
#include <tensorflow/lite/c/c_api_experimental.h>
#endif

/// user_data_ptr is a pointer to a TfLiteErrorReporterUserData
static void
tflite_error_callback(void* user_data_ptr, const char* format, va_list args);

tl::expected<std::unique_ptr<TfLiteRuntime>, std::string> TfLiteRuntime::create(
	std::vector<int8_t>&& model_data,
	std::string_view gpu_delegate_serialization_dir,
	std::string_view model_token,
	TfLiteLogWarningCallback log_warning_callback,
	TfLiteLogErrorCallback log_error_callback
) {
	PROFILE_DEPTH_SCOPE("Initialize TfLiteRuntime")

	std::unique_ptr<TfLiteRuntime> runtime(new TfLiteRuntime(
		std::move(model_data),
		TfLiteErrorReporterUserData(log_warning_callback, log_error_callback)
	));

	runtime->model = {
		TfLiteModelCreate(
			runtime->model_data.data(), runtime->model_data.size()
		),
		TfLiteModelDelete
	};

	TfLiteInterpreterOptions* interpreter_options_without_gpu_delegate =
		TfLiteInterpreterOptionsCreate();
	TfLiteInterpreterOptionsSetErrorReporter(
		interpreter_options_without_gpu_delegate, tflite_error_callback,
		&runtime->error_reporter_user_data
	);
	TfLiteInterpreterOptionsSetNumThreads(
		interpreter_options_without_gpu_delegate, 4
	);

	TfLiteInterpreterOptions* interpreter_options_with_gpu_delegate =
		TfLiteInterpreterOptionsCopy(interpreter_options_without_gpu_delegate);
	runtime->gpu_delegate =
		create_gpu_delegate(gpu_delegate_serialization_dir, model_token);
	TfLiteInterpreterOptionsAddDelegate(
		interpreter_options_with_gpu_delegate, runtime->gpu_delegate.get()
	);

	// first try to create interpreter with gpu delegate
	runtime->interpreter = {
		TfLiteInterpreterCreate(
			runtime->model.get(), interpreter_options_with_gpu_delegate
		),
		TfLiteInterpreterDelete
	};

	if (runtime->interpreter == nullptr) {
		// trying to create interpreter again, just without gpu delegate
		log_warning_callback(
			"GPU Delegate is not supported, falling back to CPU only mode"
		);
		runtime->interpreter = {
			TfLiteInterpreterCreate(
				runtime->model.get(), interpreter_options_without_gpu_delegate
			),
			TfLiteInterpreterDelete
		};
		if (runtime->interpreter == nullptr) {
			return tl::unexpected(
				"failed to create interpreter: with and without gpu delegate"
			);
		}
		runtime->interpreter_options = {
			interpreter_options_without_gpu_delegate,
			TfLiteInterpreterOptionsDelete
		};
		runtime->gpu_delegate.reset();
	} else {
		runtime->interpreter_options = {
			interpreter_options_with_gpu_delegate,
			TfLiteInterpreterOptionsDelete
		};
	}

	const TfLiteStatus allocate_tensors_status =
		TfLiteInterpreterAllocateTensors(runtime->interpreter.get());
	if (allocate_tensors_status != kTfLiteOk) {
		return tl::unexpected_fmt(
			"failed to allocate tensors: {}",
			format_tflite_status(allocate_tensors_status)
		);
	}

	return runtime;
}

TfLiteRuntime::~TfLiteRuntime() {
	PROFILE_DEPTH_SCOPE("Shutdown TfLiteRuntime")

	interpreter.reset();
	gpu_delegate.reset();
	interpreter_options.reset();
	model.reset();
}

tl::expected<void, std::string> TfLiteRuntime::invoke() {
	PROFILE_DEPTH_SCOPE("Invoking of model")

	const TfLiteStatus status = TfLiteInterpreterInvoke(interpreter.get());
	if (status == kTfLiteOk)
		return {};
	return tl::unexpected_fmt(
		"failed to invoke interpreter: {}", format_tflite_status(status)
	);
}

tl::expected<void, std::string> TfLiteRuntime::load_nonquantized_input(
	std::span<const std::byte> input_bytes,
	TfLiteTensor* input_tensor,
	TfLiteType input_type
) {
	if (input_tensor->type != input_type) {
		return tl::unexpected_fmt(
			"invalid input type of {}, expected {}",
			format_tflite_type(input_type),
			format_tflite_type(input_tensor->type)
		);
	}

	const TfLiteStatus copy_from_buffer_status = TfLiteTensorCopyFromBuffer(
		input_tensor, input_bytes.data(), input_bytes.size_bytes()
	);

	if (copy_from_buffer_status == kTfLiteOk)
		return {};

	return tl::unexpected_fmt(
		"failed to copy input buffer to tensor: {}",
		format_tflite_status(copy_from_buffer_status)
	);
}

tl::expected<void, std::string> TfLiteRuntime::read_nonquantized_output(
	std::span<std::byte> output_bytes,
	const TfLiteTensor* output_tensor,
	TfLiteType output_type
) {
	if (output_tensor->type != output_type) {
		return tl::unexpected_fmt(
			"invalid output type of {}, expected {}",
			format_tflite_type(output_type),
			format_tflite_type(output_tensor->type)
		);
	}

	const TfLiteStatus copy_from_buffer_status = TfLiteTensorCopyToBuffer(
		output_tensor, output_bytes.data(), output_bytes.size_bytes()
	);

	if (copy_from_buffer_status == kTfLiteOk)
		return {};

	return tl::unexpected_fmt(
		"failed to copy from tensor to buffer: {}",
		format_tflite_status(copy_from_buffer_status)
	);
}

void tflite_error_callback(
	void* user_data_ptr,
	const char* format,
	va_list args
) {
	const auto* user_data =
		static_cast<TfLiteErrorReporterUserData*>(user_data_ptr);

	// c style va_list args is necessary as its required by the tflite c api for
	// error reporting
	// NOLINTBEGIN(cppcoreguidelines-pro-type-vararg,
	// cppcoreguidelines-pro-bounds-array-to-pointer-decay)
	va_list args_copy;
	va_copy(args_copy, args);

	const int formatted_error_msg_length =
		std::vsnprintf(nullptr, 0, format, args_copy);
	std::vector<char> formatted_error_msg_buffer;
	formatted_error_msg_buffer.resize(formatted_error_msg_length + 1);
	std::vsnprintf(
		formatted_error_msg_buffer.data(), formatted_error_msg_buffer.size(),
		format, args
	);
	const std::string formatted_error_msg(formatted_error_msg_buffer.data());
	// NOLINTEND(cppcoreguidelines-pro-type-vararg,
	// cppcoreguidelines-pro-bounds-array-to-pointer-decay)

	user_data->log_error_callback(
		std::format("[TfLiteRuntime Error] {}", formatted_error_msg)
	);
}