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

tl::expected<std::unique_ptr<TfLiteRuntime>, TfLiteCreateRuntimeError>
TfLiteRuntime::create_impl(
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
) {
	PROFILE_SCOPE("Initialize TfLiteRuntime", profiling_frame)

	std::unique_ptr<TfLiteRuntime> runtime(new TfLiteRuntime(
		std::move(model_data), model_input_format, model_output_format,
		std::move(input_operators), std::move(output_operators),
		TfLiteReporterUserData(log_warning_callback, log_error_callback),
		profiling_frame
	));

	runtime->model = {
		TfLiteModelCreate(
			runtime->model_data.data(), runtime->model_data.size()
		),
		TfLiteModelDelete
	};

	std::unique_ptr<
		TfLiteInterpreterOptions, decltype(&TfLiteInterpreterOptionsDelete)>
		interpreter_options_without_gpu_delegate = {
			TfLiteInterpreterOptionsCreate(), TfLiteInterpreterOptionsDelete
		};
	TfLiteInterpreterOptionsSetErrorReporter(
		interpreter_options_without_gpu_delegate.get(), tflite_error_callback,
		&runtime->reporter_user_data
	);
	TfLiteInterpreterOptionsSetNumThreads(
		interpreter_options_without_gpu_delegate.get(), 4
	);

	std::unique_ptr<
		TfLiteInterpreterOptions, decltype(&TfLiteInterpreterOptionsDelete)>
		interpreter_options_with_gpu_delegate = {
			TfLiteInterpreterOptionsCopy(
				interpreter_options_without_gpu_delegate.get()
			),
			TfLiteInterpreterOptionsDelete
		};
	runtime->gpu_delegate =
		create_gpu_delegate(gpu_delegate_serialization_dir, model_token, profiling_frame);
	TfLiteInterpreterOptionsAddDelegate(
		interpreter_options_with_gpu_delegate.get(), runtime->gpu_delegate.get()
	);

	// first try to create interpreter with gpu delegate
	runtime->interpreter = {
		TfLiteInterpreterCreate(
			runtime->model.get(), interpreter_options_with_gpu_delegate.get()
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
				runtime->model.get(),
				interpreter_options_without_gpu_delegate.get()
			),
			TfLiteInterpreterDelete
		};
		if (runtime->interpreter == nullptr) {
			return tl::unexpected(TfLiteCreateInterpreterError());
		}
		runtime->interpreter_options =
			std::move(interpreter_options_without_gpu_delegate);
		runtime->gpu_delegate.reset();
	} else {
		runtime->interpreter_options =
			std::move(interpreter_options_with_gpu_delegate);
	}

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

std::optional<TfLiteRunInferenceError> TfLiteRuntime::run_inference(
	std::span<float> input,
	FloatTensorFormat input_format,
	std::span<float> output,
	FloatTensorFormat expected_output_format
) {
	PROFILE_FUNCTION(profiling_frame)

	FloatTensorFormat current_input_format = input_format;

	{
		PROFILE_SCOPE("Preprocessing input using operators", profiling_frame)

		for (auto& input_operator : input_operators) {
			const FloatTensorFormat operator_input_format =
				input_operator->get_input_format();

			if (current_input_format != operator_input_format) {
				return InvalidInputFormatForOperator{
					.provided = current_input_format,
					.expected = operator_input_format
				};
			}

			if (const auto error = input_operator->execute(input))
				return error;

			current_input_format = input_operator->get_output_format();
		}
	}

	if (current_input_format != model_input_format) {
		return InvalidInputFormatForModel{
			.provided = current_input_format, .expected = model_input_format
		};
	}

	if (const auto load_input_error = load_input(input))
		return load_input_error;

	if (const auto invoke_error = invoke())
		return invoke_error;

	if (const auto read_output_error = read_output(output))
		return read_output_error;

	FloatTensorFormat current_output_format = model_output_format;

	{
		PROFILE_SCOPE("Postprocessing output using operators", profiling_frame)

		for (auto& output_operator : output_operators) {
			const FloatTensorFormat operator_input_format =
				output_operator->get_input_format();

			if (current_output_format != operator_input_format) {
				return InvalidInputFormatForOperator{
					.provided = current_input_format,
					.expected = operator_input_format
				};
			}

			if (const auto error = output_operator->execute(output))
				return error;

			current_output_format = output_operator->get_output_format();
		}

		if (current_output_format != expected_output_format) {
			return UnexpectedOperatorOutputFormat{
				.operator_output = current_output_format,
				.expected_output = expected_output_format
			};
		}
	}

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

	return read_floats_from_output_tensor(output_tensor, output, profiling_frame);
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