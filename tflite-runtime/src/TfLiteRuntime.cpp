#include "tflite-runtime/TfLiteRuntime.hpp"
#include "tflite-runtime/TfLiteUtils.hpp"

#include <format>

#if TFLITE_RUNTIME_USE_PREBUILT_TFLITE
#include <tflite/c/c_api_experimental.h>
#else
#include <tensorflow/lite/c/c_api_experimental.h>
#endif

/// user_data_ptr is a pointer to a TfLiteErrorReporterUserData
static void
tflite_error_callback(void* user_data_ptr, const char* format, va_list args);

tl::expected<std::unique_ptr<TfLiteRuntime>, ErrorMsg> TfLiteRuntime::create(
	std::span<const int8_t> model_data,
	std::string_view delegate_serialization_dir,
	std::string_view model_token,
	TfLiteLogWarningCallback log_warning_callback,
	TfLiteLogErrorCallback log_error_callback,
	NpuConfiguration npu_config,
	bool enable_npu,
	std::string_view skel_library_dir
) {
	std::unique_ptr<TfLiteRuntime> runtime(new TfLiteRuntime(
		model_data,
		TfLiteReporterUserData(log_warning_callback, log_error_callback)
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
		interpreter_options_with_gpu_and_npu_delegate = {
			TfLiteInterpreterOptionsCopy(
				interpreter_options_without_gpu_delegate.get()
			),
			TfLiteInterpreterOptionsDelete
		};

	if (enable_npu) {
		runtime->npu_delegate = create_qnn_npu_delegate(
			delegate_serialization_dir, model_token, npu_config,
			skel_library_dir
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

	runtime->gpu_delegate =
		create_gpu_delegate(delegate_serialization_dir, model_token);
	TfLiteInterpreterOptionsAddDelegate(
		interpreter_options_with_gpu_and_npu_delegate.get(),
		runtime->gpu_delegate.get()
	);

	// first try to create interpreter with gpu delegate
	runtime->interpreter = {
		TfLiteInterpreterCreate(
			runtime->model.get(),
			interpreter_options_with_gpu_and_npu_delegate.get()
		),
		TfLiteInterpreterDelete
	};

	if (runtime->interpreter == nullptr) {
		// trying to create interpreter again, just without gpu delegate
		log_warning_callback(
			"GPU or NPU Delegate is not supported, falling back to CPU only "
			"mode"
		);
		runtime->interpreter = {
			TfLiteInterpreterCreate(
				runtime->model.get(),
				interpreter_options_without_gpu_delegate.get()
			),
			TfLiteInterpreterDelete
		};
		if (runtime->interpreter == nullptr) {
			return tl::unexpected("failed to create tflite interpreter");
		}
		runtime->interpreter_options =
			std::move(interpreter_options_without_gpu_delegate);
		runtime->gpu_delegate.reset();
		runtime->npu_delegate.reset();
	} else {
		runtime->interpreter_options =
			std::move(interpreter_options_with_gpu_and_npu_delegate);
	}

	const TfLiteStatus allocate_tensors_status =
		TfLiteInterpreterAllocateTensors(runtime->interpreter.get());
	if (allocate_tensors_status != kTfLiteOk) {
		return tl::unexpected(
			std::format(
				"failed to allocate tflite tensors: {}",
				format_tflite_status(allocate_tensors_status)
			)
		);
	}

	return runtime;
}

TfLiteRuntime::~TfLiteRuntime() {
	interpreter.reset();
	gpu_delegate.reset();
	npu_delegate.reset();
	interpreter_options.reset();
	model.reset();
}

std::optional<ErrorMsg> TfLiteRuntime::invoke() {
	const TfLiteStatus status = TfLiteInterpreterInvoke(interpreter.get());
	if (status == kTfLiteOk)
		return std::nullopt;
	return std::format(
		"failed to invoke tflite interpreter: {}", format_tflite_status(status)
	);
}

std::optional<ErrorMsg>
TfLiteRuntime::run_inference(std::span<float> input, std::span<float> output) {
	if (const auto error = load_input(input))
		return error;

	if (const auto error = invoke())
		return error;

	if (const auto error = read_output(output))
		return error;

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

std::optional<ErrorMsg>
TfLiteRuntime::load_input(std::span<const float> input) {
	TfLiteTensor* input_tensor =
		TfLiteInterpreterGetInputTensor(interpreter.get(), 0);

	return load_input_tensor_with_floats(input_tensor, input);
}

std::optional<ErrorMsg> TfLiteRuntime::read_output(std::span<float> output) {
	const TfLiteTensor* output_tensor =
		TfLiteInterpreterGetOutputTensor(interpreter.get(), 0);

	return read_floats_from_output_tensor(output_tensor, output);
}

void tflite_error_callback(
	void* user_data_ptr,
	const char* format,
	va_list args
) {
	const auto* user_data = static_cast<TfLiteReporterUserData*>(user_data_ptr);

	// c style va_list args is necessary as its required by the tflite c api
	// for error reporting NOLINTBEGIN(cppcoreguidelines-pro-type-vararg,
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
	// NOLINTEND(cppcoreguidelines-pro-type-vararg,
	// cppcoreguidelines-pro-bounds-array-to-pointer-decay)

	const std::string formatted_str = std::format(
		"[TfLiteRuntime Error] {}", formatted_error_msg_buffer.data()
	);

	user_data->log_error_callback(formatted_str.c_str());
}
