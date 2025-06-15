#include "EyeAICore/onnx/OnnxRuntime.hpp"
#include "EyeAICore/onnx/OnnxUtils.hpp"
#include "EyeAICore/utils/Errors.hpp"
#include "EyeAICore/utils/Profiling.hpp"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"

#include <cpu_provider_factory.h>
#include <format>
#include <nnapi_provider_factory.h>

static void onnx_logging_callback(
	void* log_callbacks_param, // see OnnxRuntime constructor
	OrtLoggingLevel severity,
	const char* category,
	const char* logid,
	const char* code_location,
	const char* message
) {
	const auto* callbacks = static_cast<OnnxLogCallbacks*>(log_callbacks_param);
	switch (severity) {
	default:
		callbacks->log_info(
			std::format(
				"[{}] [{}] {} ({})", category, logid, message, code_location
			)
		);
		break;
	case ORT_LOGGING_LEVEL_ERROR:
	case ORT_LOGGING_LEVEL_FATAL:
		callbacks->log_error(
			std::format(
				"[{}] [{}] {} ({})", category, logid, message, code_location
			)
		);
		break;
	}
}

tl::expected<std::unique_ptr<OnnxRuntime>, std::string> OnnxRuntime::create(
	std::span<const std::byte> model_data,
	OnnxLogCallbacks&& log_callbacks
) {
	PROFILE_DEPTH_SCOPE("Init OnnxRuntime")

	std::unique_ptr<OnnxRuntime> runtime(
		new OnnxRuntime(std::move(log_callbacks))
	);

	Ort::Env env(
		ORT_LOGGING_LEVEL_WARNING, "Default", onnx_logging_callback,
		&runtime->log_callbacks
	);

	auto session_options = Ort::SessionOptions();
	session_options.SetInterOpNumThreads(4);
	session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

	const auto append_cpu_status = Ort::Status(
		OrtSessionOptionsAppendExecutionProvider_CPU(session_options, true)
	);
	if (!append_cpu_status.IsOK()) {
		return tl::unexpected_fmt(
			"failed to append cpu execution provider: {}: {}",
			format_ort_error_code(append_cpu_status.GetErrorCode()),
			append_cpu_status.GetErrorMessage()
		);
	}

	// TODO: test out what impact NNAPI_FLAG_USE_FP16 has
	constexpr uint32_t NNAPI_FLAGS =
		NNAPI_FLAG_USE_FP16; // | NNAPI_FLAG_CPU_DISABLED;

	const auto append_nnapi_status =
		Ort::Status(OrtSessionOptionsAppendExecutionProvider_Nnapi(
			session_options, NNAPI_FLAGS
		));
	if (!append_nnapi_status.IsOK()) {
		return tl::unexpected_fmt(
			"failed to append nnapi execution provider: {}: {}",
			format_ort_error_code(append_nnapi_status.GetErrorCode()),
			append_nnapi_status.GetErrorMessage()
		);
	}

	runtime->session = Ort::Session(
		runtime->env, model_data.data(), model_data.size_bytes(),
		session_options
	);

	const auto input_names = runtime->session.GetInputNames();
	if (input_names.size() != 1)
		return tl::unexpected_fmt(
			"invalid model input count: {}", input_names.size()
		);
	runtime->input_name = input_names[0];

	const auto input_type_and_shape_info =
		runtime->session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
	runtime->input_shape = input_type_and_shape_info.GetShape();
	runtime->input_type = input_type_and_shape_info.GetElementType();

	const auto output_names = runtime->session.GetOutputNames();
	if (output_names.size() != 1)
		return tl::unexpected_fmt(
			"invalid model output count: {}", output_names.size()
		);
	runtime->output_name = output_names[0];

	const auto output_type_and_shape_info =
		runtime->session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
	runtime->output_shape = output_type_and_shape_info.GetShape();
	runtime->output_type = output_type_and_shape_info.GetElementType();

	runtime->memory_info =
		Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

	return runtime;
}

void OnnxRuntime::run_inference_raw(
	std::span<std::byte> input_data,
	std::span<std::byte> output_data
) {
	PROFILE_DEPTH_SCOPE("Run Inference")

	Ort::Value input_tensor;
	Ort::Value output_tensor;

	{
		PROFILE_DEPTH_SCOPE("Preparing Inference")

		input_tensor = Ort::Value::CreateTensor(
			memory_info, input_data.data(), input_data.size_bytes(),
			input_shape.data(), input_shape.size(), input_type
		);
		output_tensor = Ort::Value::CreateTensor(
			memory_info, output_data.data(), output_data.size_bytes(),
			output_shape.data(), output_shape.size(), output_type
		);
	}

	{
		PROFILE_DEPTH_SCOPE("Invoking model")

		const char* input_names{input_name.data()};
		const char* output_names{output_name.data()};
		const Ort::RunOptions run_options;

		session.Run(
			run_options, &input_names, &input_tensor, 1, &output_names,
			&output_tensor, 1
		);
	}
}