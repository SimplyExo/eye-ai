#include "EyeAICore/tflite/TfLiteRuntime.hpp"
#include "EyeAICore/tflite/TfLiteUtils.hpp"
#include "EyeAICore/utils/Profiling.hpp"

#include <format>
#include <tflite/c/c_api_experimental.h>

static void
tflite_error_callback(void* /*user_data*/, const char* format, va_list args);

tl::expected<std::unique_ptr<TfLiteRuntime>, std::string> TfLiteRuntime::create(
	std::span<const int8_t> model_data,
	std::string_view gpu_delegate_serialization_dir,
	std::string_view model_token,
	TfLiteLogErrorCallback log_error_callback
) {
	PROFILE_DEPTH_SCOPE("Initialize TfLiteRuntime")

	std::unique_ptr<TfLiteRuntime> runtime(new TfLiteRuntime(log_error_callback)
	);

	runtime->model = TfLiteModelCreate(model_data.data(), model_data.size());

	runtime->interpreter_options = TfLiteInterpreterOptionsCreate();
	TfLiteInterpreterOptionsSetErrorReporter(
		runtime->interpreter_options, tflite_error_callback,
		reinterpret_cast<void*>(log_error_callback)
	);
	TfLiteInterpreterOptionsSetNumThreads(runtime->interpreter_options, 4);

	runtime->gpu_delegate =
		create_gpu_delegate(gpu_delegate_serialization_dir, model_token);
	TfLiteInterpreterOptionsAddDelegate(
		runtime->interpreter_options, runtime->gpu_delegate
	);

	runtime->interpreter =
		TfLiteInterpreterCreate(runtime->model, runtime->interpreter_options);

	const TfLiteStatus allocate_tensors_status =
		TfLiteInterpreterAllocateTensors(runtime->interpreter);
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

	TfLiteInterpreterDelete(interpreter);
	if (gpu_delegate != nullptr)
		delete_gpu_delegate(gpu_delegate);
	TfLiteInterpreterOptionsDelete(interpreter_options);
	TfLiteModelDelete(model);
}

tl::expected<void, std::string> TfLiteRuntime::invoke() {
	PROFILE_DEPTH_SCOPE("Invoking of model")

	const TfLiteStatus status = TfLiteInterpreterInvoke(interpreter);
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

void tflite_error_callback(void* user_data, const char* format, va_list args) {
	// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
	const auto log_error_calback =
		reinterpret_cast<TfLiteLogErrorCallback>(user_data);
	// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

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

	log_error_calback(
		std::format("[TfLiteRuntime Error] {}", formatted_error_msg)
	);
}