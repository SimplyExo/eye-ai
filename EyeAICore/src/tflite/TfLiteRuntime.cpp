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

/**
 * tries to create a tflite interpreter with the most hardware delegates enabled,
 * so npu + gpu delegate if supported, else gpu if supported, or for fallback cpu only
 */
static tl::expected<TfLiteInterpreterPtr, TfLiteCreateInterpreterError>
try_create_interpreter(
	const TfLiteModelPtr& model,
	TfLiteInterpreterOptionsPtr cpu_only_options,
	TfLiteInterpreterOptionsPtr gpu_delegate_options,
	TfLiteInterpreterOptionsPtr gpu_and_npu_delegate_options,
	TfLiteInterpreterOptionsPtr& out_used_options,
	TfLiteLogWarningCallback log_warning_callback
) {
	TfLiteInterpreterPtr interpreter = {
		TfLiteInterpreterCreate(
			model.get(), gpu_and_npu_delegate_options.get()
		),
		TfLiteInterpreterDelete
	};
	if (interpreter) {
		log_warning_callback(
			"TfLite Interpreter created with NPU and GPU delegate support!"
		);
		out_used_options = std::move(gpu_and_npu_delegate_options);
		return interpreter;
	}

	interpreter = {
		TfLiteInterpreterCreate(model.get(), gpu_delegate_options.get()),
		TfLiteInterpreterDelete
	};
	if (interpreter) {
		log_warning_callback(
			"TfLite Interpreter created with GPU delegate support!"
		);
		out_used_options = std::move(gpu_delegate_options);
		return interpreter;
	}

	interpreter = {
		TfLiteInterpreterCreate(model.get(), cpu_only_options.get()),
		TfLiteInterpreterDelete
	};
	if (interpreter) {
		log_warning_callback(
			"TfLite Interpreter created with no NPU or GPU delegate support, "
			"cpu only mode!"
		);
		out_used_options = std::move(cpu_only_options);
		return interpreter;
	}
	return tl::unexpected(TfLiteCreateInterpreterError());
}

tl::expected<std::unique_ptr<TfLiteRuntime>, TfLiteCreateRuntimeError>
TfLiteRuntime::create(
	std::vector<int8_t>&& model_data,
	std::string_view delegate_serialization_dir,
	std::string_view model_token,
	FloatTensorFormat model_input_format,
	FloatTensorFormat model_output_format,
	TfLiteLogWarningCallback log_warning_callback,
	TfLiteLogErrorCallback log_error_callback,
	ProfilingFrame& profiling_frame,
	bool enable_npu,
	NpuConfiguration npu_config,
	std::string npu_skel_directory
) {
	PROFILE_SCOPE("Initialize TfLiteRuntime", profiling_frame)

	std::unique_ptr<TfLiteRuntime> runtime(new TfLiteRuntime(
		std::move(model_data), model_input_format, model_output_format,
		TfLiteReporterUserData(log_warning_callback, log_error_callback),
		profiling_frame, std::move(npu_skel_directory)
	));

	runtime->model = {
		TfLiteModelCreate(
			runtime->model_data.data(), runtime->model_data.size()
		),
		TfLiteModelDelete
	};

	TfLiteInterpreterOptionsPtr interpreter_options_cpu_only = {
		TfLiteInterpreterOptionsCreate(), TfLiteInterpreterOptionsDelete
	};
	TfLiteInterpreterOptionsSetErrorReporter(
		interpreter_options_cpu_only.get(), tflite_error_callback,
		&runtime->reporter_user_data
	);
	TfLiteInterpreterOptionsSetNumThreads(
		interpreter_options_cpu_only.get(), 4
	);

	// GPU Delegate
	TfLiteInterpreterOptionsPtr interpreter_options_with_gpu_delegate = {
		TfLiteInterpreterOptionsCopy(interpreter_options_cpu_only.get()),
		TfLiteInterpreterOptionsDelete
	};
	runtime->gpu_delegate = create_gpu_delegate(
		delegate_serialization_dir, model_token, profiling_frame
	);
	TfLiteInterpreterOptionsAddDelegate(
		interpreter_options_with_gpu_delegate.get(), runtime->gpu_delegate.get()
	);

	// NPU Delegate
	TfLiteInterpreterOptionsPtr
		interpreter_options_with_gpu_and_npu_delegate = {
			TfLiteInterpreterOptionsCopy(
				interpreter_options_cpu_only.get()
			),
			TfLiteInterpreterOptionsDelete
		};
	TfLiteInterpreterOptionsAddDelegate(
		interpreter_options_with_gpu_and_npu_delegate.get(), runtime->gpu_delegate.get()
	);
	if (enable_npu) {
		runtime->npu_delegate = create_qnn_npu_delegate(
			delegate_serialization_dir, model_token, npu_config,
			runtime->npu_skel_directory
		);
		if (runtime->npu_delegate == nullptr) {
			log_warning_callback("No QNN NPU delegate was created!");
		} else {
			log_warning_callback("QNN NPU delegate was created!");
			TfLiteInterpreterOptionsAddDelegate(
				interpreter_options_with_gpu_and_npu_delegate.get(),
				runtime->npu_delegate.get()
			);
		}
	}

	auto interpreter_result = try_create_interpreter(
		runtime->model,
		std::move(interpreter_options_cpu_only),
		std::move(interpreter_options_with_gpu_delegate),
		std::move(interpreter_options_with_gpu_and_npu_delegate),
		runtime->interpreter_options,
		log_warning_callback
	);
	if (!interpreter_result.has_value())
		return tl::unexpected(interpreter_result.error());

	runtime->interpreter = std::move(interpreter_result.value());

	const TfLiteStatus allocate_tensors_status =
		TfLiteInterpreterAllocateTensors(runtime->interpreter.get());
	if (allocate_tensors_status != kTfLiteOk) {
		return tl::unexpected(
			TfLiteAllocateTensorsError{.status = allocate_tensors_status}
		);
	}

	return runtime;
}

TfLiteRuntime::~TfLiteRuntime() {
	PROFILE_SCOPE("Shutdown TfLiteRuntime", profiling_frame)

	interpreter.reset();
	gpu_delegate.reset();
	npu_delegate.reset();
	interpreter_options.reset();
	model.reset();
}

std::optional<TfLiteInvokeInterpreterError> TfLiteRuntime::invoke() {
	PROFILE_SCOPE("Invoking of model", profiling_frame)

	const TfLiteStatus status = TfLiteInterpreterInvoke(interpreter.get());
	if (status == kTfLiteOk)
		return std::nullopt;
	return TfLiteInvokeInterpreterError{status};
}

std::optional<TfLiteRunInferenceError>
TfLiteRuntime::run_inference(std::span<float> input, std::span<float> output) {
	PROFILE_FUNCTION(profiling_frame)

	if (const auto load_input_error = load_input(input))
		return load_input_error;

	if (const auto invoke_error = invoke())
		return invoke_error;

	if (const auto read_output_error = read_output(output))
		return read_output_error;

	return std::nullopt;
}

std::span<const int> TfLiteRuntime::get_input_shape() const {
	const TfLiteTensor* input_tensor =
		TfLiteInterpreterGetInputTensor(interpreter.get(), 0);

	return get_tensor_shape(input_tensor);
}

std::span<const int> TfLiteRuntime::get_output_shape() const {
	const TfLiteTensor* output_tensor =
		TfLiteInterpreterGetOutputTensor(interpreter.get(), 0);

	return get_tensor_shape(output_tensor);
}

std::optional<TfLiteLoadInputError>
TfLiteRuntime::load_input(std::span<const float> input) {
	PROFILE_SCOPE("Loading input", profiling_frame)

	TfLiteTensor* input_tensor =
		TfLiteInterpreterGetInputTensor(interpreter.get(), 0);

	return load_input_tensor_with_floats(input_tensor, input, profiling_frame);
}

std::optional<TfLiteReadOutputError>
TfLiteRuntime::read_output(std::span<float> output) {
	PROFILE_SCOPE("Reading output", profiling_frame)

	const TfLiteTensor* output_tensor =
		TfLiteInterpreterGetOutputTensor(interpreter.get(), 0);

	return read_floats_from_output_tensor(
		output_tensor, output, profiling_frame
	);
}

void tflite_error_callback(
	void* user_data_ptr,
	const char* format,
	va_list args
) {
	const auto* user_data = static_cast<TfLiteReporterUserData*>(user_data_ptr);

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

std::string TfLiteCreateInterpreterError::to_string() {
	return "failed to create TfLite Interpreter (with and without gpu "
		   "delegate)";
}

std::string TfLiteAllocateTensorsError::to_string() const {
	return std::format(
		"failed to allocate tflite tensors: {}", format_tflite_status(status)
	);
}

std::string TfLiteInvokeInterpreterError::to_string() const {
	return std::format(
		"failed to invoke tflite interpreter: {}", format_tflite_status(status)
	);
}