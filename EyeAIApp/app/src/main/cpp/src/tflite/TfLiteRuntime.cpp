#include "TfLiteRuntime.hpp"
#include "tflite/TfLiteUtils.hpp"
#include "utils/Profiling.hpp"

#include "tflite/c/common.h"
#include <cassert>

static void
tflite_error_callback(void* /*user_data*/, const char* format, va_list args);

// NOLINTBEGIN(modernize-use-default-member-init,
// cppcoreguidelines-prefer-member-initializer)
TfLiteRuntime::TfLiteRuntime(
	std::span<const int8_t> model_data,
	std::string_view gpu_delegate_serialization_dir,
	std::string_view model_token
)
	: model(nullptr), interpreter(nullptr), interpreter_options(nullptr),
	  gpu_delegate(nullptr) {

	PROFILE_DEPTH_SCOPE("Initialize TfLiteRuntime")

	model = TfLiteModelCreate(model_data.data(), model_data.size());

	interpreter_options = TfLiteInterpreterOptionsCreate();
	TfLiteInterpreterOptionsSetErrorReporter(
		interpreter_options, tflite_error_callback, nullptr
	);
	TfLiteInterpreterOptionsSetNumThreads(interpreter_options, 4);

	gpu_delegate =
		create_gpu_delegate(gpu_delegate_serialization_dir, model_token);
	TfLiteInterpreterOptionsAddDelegate(interpreter_options, gpu_delegate);

	interpreter = TfLiteInterpreterCreate(model, interpreter_options);

	throw_on_tflite_status(
		TfLiteInterpreterAllocateTensors(interpreter),
		"failed to allocate tensors"
	);
}
// NOLINTEND(modernize-use-default-member-init,
// cppcoreguidelines-prefer-member-initializer)

TfLiteRuntime::~TfLiteRuntime() {
	PROFILE_DEPTH_SCOPE("Shutdown TfLiteRuntime")

	TfLiteInterpreterDelete(interpreter);
	if (gpu_delegate != nullptr)
		TfLiteGpuDelegateV2Delete(gpu_delegate);
	TfLiteInterpreterOptionsDelete(interpreter_options);
	TfLiteModelDelete(model);
}

void TfLiteRuntime::load_nonquantized_input(
	std::span<const std::byte> input_bytes,
	TfLiteTensor* input_tensor,
	TfLiteType input_type
) {
	if (input_tensor->type != input_type)
		throw WrongTypeException(input_tensor->type, input_type);

	throw_on_tflite_status(
		TfLiteTensorCopyFromBuffer(
			input_tensor, input_bytes.data(), input_bytes.size_bytes()
		),
		"failed to load input from tensor"
	);
}

void TfLiteRuntime::read_nonquantized_output(
	std::span<std::byte> output_bytes,
	const TfLiteTensor* output_tensor,
	TfLiteType output_type
) {
	if (output_tensor->type != output_type)
		throw WrongTypeException(output_tensor->type, output_type);

	throw_on_tflite_status(
		TfLiteTensorCopyToBuffer(
			output_tensor, output_bytes.data(), output_bytes.size_bytes()
		),
		"failed to read output from tensor"
	);
}

void tflite_error_callback(
	void* /*user_data*/,
	const char* format,
	va_list args
) {
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

	LOG_ERROR("[TfLiteRuntime Error] {}", formatted_error_msg.c_str());
}