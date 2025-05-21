#include "OnnxRuntime.hpp"
#include "onnxruntime_c_api.h"
#include "onnxruntime_cxx_api.h"
#include "utils/Exceptions.hpp"
#include "utils/Log.hpp"
#include "utils/Profiling.hpp"

#include <cpu_provider_factory.h>
#include <nnapi_provider_factory.h>

static void onnx_logging_callback(
	void* /*param*/,
	OrtLoggingLevel severity,
	const char* category,
	const char* logid,
	const char* code_location,
	const char* message
) {
	switch (severity) {
	default:
		LOG_INFO(
			"[OnnxRuntime] [{}] [{}] {} ({})", category, logid, message,
			code_location
		);
		break;
	case ORT_LOGGING_LEVEL_ERROR:
	case ORT_LOGGING_LEVEL_FATAL:
		LOG_ERROR(
			"[OnnxRuntime] [{}] [{}] {} ({})", category, logid, message,
			code_location
		);
		break;
	}
}

OnnxRuntime::OnnxRuntime(std::span<const std::byte> model_data) {
	PROFILE_DEPTH_SCOPE("Init OnnxRuntime")

	env = Ort::Env(
		ORT_LOGGING_LEVEL_WARNING, "Default", onnx_logging_callback, nullptr
	);

	auto session_options = Ort::SessionOptions();
	session_options.SetInterOpNumThreads(4);
	session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

	throw_on_onnx_status(
		Ort::Status(OrtSessionOptionsAppendExecutionProvider_CPU(
			session_options, (int)true
		))
	);

	// TODO: test out what impact NNAPI_FLAG_USE_FP16 has
	const uint32_t nnapi_flags =
		NNAPI_FLAG_USE_FP16; // | NNAPI_FLAG_CPU_DISABLED;

	throw_on_onnx_status(
		Ort::Status(OrtSessionOptionsAppendExecutionProvider_Nnapi(
			session_options, nnapi_flags
		))
	);

	session = Ort::Session(
		env, model_data.data(), model_data.size_bytes(), session_options
	);

	const auto input_names = session.GetInputNames();
	if (input_names.size() != 1)
		throw OnnxInvalidInputCount(input_names.size());
	input_name = input_names[0];

	const auto input_type_and_shape_info =
		session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
	input_shape = input_type_and_shape_info.GetShape();
	input_type = input_type_and_shape_info.GetElementType();

	const auto output_names = session.GetOutputNames();
	if (output_names.size() != 1)
		throw OnnxInvalidOutputCount(output_names.size());
	output_name = output_names[0];

	const auto output_type_and_shape_info =
		session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
	output_shape = output_type_and_shape_info.GetShape();
	output_type = output_type_and_shape_info.GetElementType();

	memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);
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