#include "EyeAICore/tflite/TfLiteRuntime.hpp"
#include "EyeAICore/tflite/TfLiteUtils.hpp"
#include "EyeAICore/utils/Errors.hpp"
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
	std::vector<std::unique_ptr<Operator>>&& input_operators,
	std::vector<std::unique_ptr<Operator>>&& output_operators,
	TfLiteLogWarningCallback log_warning_callback,
	TfLiteLogErrorCallback log_error_callback
) {
	PROFILE_DEPTH_SCOPE("Initialize TfLiteRuntime")

	std::unique_ptr<TfLiteRuntime> runtime(new TfLiteRuntime(
		std::move(model_data), std::move(input_operators),
		std::move(output_operators),
		TfLiteErrorReporterUserData(log_warning_callback, log_error_callback)
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
		&runtime->error_reporter_user_data
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
		create_gpu_delegate(gpu_delegate_serialization_dir, model_token);
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
			return tl::unexpected(
				"failed to create interpreter: with and without gpu delegate"
			);
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

tl::expected<void, std::string>
TfLiteRuntime::run_inference(std::span<float> input, std::span<float> output) {
	PROFILE_DEPTH_FUNCTION()

	{
		PROFILE_DEPTH_SCOPE("Preprocessing input using operators")

		for (auto& input_operator : input_operators) {
			const auto result = input_operator->execute(input);
			if (!result.has_value())
				return result;
		}
	}

	// clang-format off
	return load_input(input)
		.and_then(
			[this]() { return invoke(); }
		)
		.and_then(
			[this, output]() { return read_output(output); }
		)
		.and_then(
			[this, output]() {
				PROFILE_DEPTH_SCOPE("Postprocessing output using operators")

				for (auto& output_operator : output_operators) {
					const auto result = output_operator->execute(output);
					if (!result.has_value())
						return result;
				}

				return tl::expected<void, std::string>{};
			}
		);
	// clang-format on
}

tl::expected<void, std::string>
TfLiteRuntime::load_input(std::span<const float> input) {
	PROFILE_DEPTH_SCOPE("Loading input")

	TfLiteTensor* input_tensor =
		TfLiteInterpreterGetInputTensor(interpreter.get(), 0);

	return load_input_tensor_with_floats(input_tensor, input);
}

tl::expected<void, std::string>
TfLiteRuntime::read_output(std::span<float> output) {
	PROFILE_DEPTH_SCOPE("Reading output")

	const TfLiteTensor* output_tensor =
		TfLiteInterpreterGetOutputTensor(interpreter.get(), 0);

	return read_floats_from_output_tensor(output_tensor, output);
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

TfLiteRuntimeBuilder::TfLiteRuntimeBuilder(
	std::vector<int8_t>&& model_data,
	std::string_view gpu_delegate_serialization_dir,
	std::string_view model_token,
	TfLiteLogWarningCallback log_warning_callback,
	TfLiteLogErrorCallback log_error_callback
)
	: model_data(std::move(model_data)),
	  gpu_delegate_serialization_dir(gpu_delegate_serialization_dir),
	  model_token(model_token), log_warning_callback(log_warning_callback),
	  log_error_callback(log_error_callback) {}

TfLiteRuntimeBuilder& TfLiteRuntimeBuilder::add_input_operator(
	std::unique_ptr<Operator>&& input_operator
) {
	input_operators.push_back(std::move(input_operator));
	return *this;
}

TfLiteRuntimeBuilder& TfLiteRuntimeBuilder::add_output_operator(
	std::unique_ptr<Operator>&& output_operator
) {
	output_operators.push_back(std::move(output_operator));
	return *this;
}

tl::expected<std::unique_ptr<TfLiteRuntime>, std::string>
TfLiteRuntimeBuilder::build() {
	return TfLiteRuntime::create(
		std::move(model_data), gpu_delegate_serialization_dir, model_token,
		std::move(input_operators), std::move(output_operators),
		log_warning_callback, log_error_callback
	);
}