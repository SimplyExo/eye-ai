#include "EyeAICore/tflite/TfLiteRuntime.hpp"
#include "EyeAICore/tflite/TfLiteUtils.hpp"
#include "EyeAICore/utils/Profiling.hpp"

#include <format>
#include <tflite/c/c_api_experimental.h>

static void
tflite_error_callback(void* /*user_data*/, const char* format, va_list args);

// NOLINTBEGIN(modernize-use-default-member-init,
// cppcoreguidelines-prefer-member-initializer)
TfLiteRuntime::TfLiteRuntime(
	std::span<const int8_t> model_data,
	std::string_view gpu_delegate_serialization_dir,
	std::string_view model_token,
	bool enable_profiling,
	TfLiteLogErrorCallback log_error_callback
)
	: model(nullptr), interpreter(nullptr), interpreter_options(nullptr),
	  gpu_delegate(nullptr), log_error_callback(log_error_callback) {

	PROFILE_DEPTH_SCOPE("Initialize TfLiteRuntime")

	model = TfLiteModelCreate(model_data.data(), model_data.size());

	interpreter_options = TfLiteInterpreterOptionsCreate();
	TfLiteInterpreterOptionsSetErrorReporter(
		interpreter_options, tflite_error_callback,
		reinterpret_cast<void*>(log_error_callback)
	);
	TfLiteInterpreterOptionsSetNumThreads(interpreter_options, 4);

	if (enable_profiling) {
		telemetry_profiler = TfLiteTelemetryProfilerStruct{
			.data = this,
			.ReportTelemetryEvent =
				[](struct TfLiteTelemetryProfilerStruct* profiler,
				   const char* event_name, uint64_t status) {
					/* unused */
				},
			.ReportTelemetryOpEvent =
				[](struct TfLiteTelemetryProfilerStruct* profiler,
				   const char* event_name, int64_t op_idx, int64_t subgraph_idx,
				   uint64_t status) { /* not used */ },
			.ReportSettings = [](struct TfLiteTelemetryProfilerStruct* profiler,
								 const char* setting_name,
								 const TfLiteTelemetrySettings* settings
							  ) { /* unused */ },
			.ReportBeginOpInvokeEvent =
				[](struct TfLiteTelemetryProfilerStruct* profiler,
				   const char* op_name, int64_t op_idx,
				   int64_t subgraph_idx) -> uint32_t {
				/* unused */
				return 0;
			},
			.ReportEndOpInvokeEvent =
				[](struct TfLiteTelemetryProfilerStruct* profiler,
				   uint32_t event_handle) { /* unused */ },
			.ReportOpInvokeEvent =
				[](struct TfLiteTelemetryProfilerStruct* profiler,
				   const char* op_name, uint64_t elapsed_time, int64_t op_idx,
				   int64_t subgraph_idx) {
					auto* runtime =
						reinterpret_cast<TfLiteRuntime*>(profiler->data);
					runtime->current_invoke_profiler_entries.emplace_back(
						op_name, std::chrono::microseconds(elapsed_time)
					);
				},
		};
		TfLiteInterpreterOptionsSetTelemetryProfiler(
			interpreter_options, &telemetry_profiler.value()
		);
	}

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
		delete_gpu_delegate(gpu_delegate);
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