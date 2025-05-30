#include "EyeAICore/onnx/OnnxUtils.hpp"
#include <format>

std::string_view format_ort_error_code(OrtErrorCode error_code) {
	switch (error_code) {
	case ORT_OK:
		return "Ok";
	case ORT_FAIL:
		return "Fail";
	case ORT_INVALID_ARGUMENT:
		return "Invalid argument";
	case ORT_NO_SUCHFILE:
		return "No such file";
	case ORT_NO_MODEL:
		return "No model";
	case ORT_ENGINE_ERROR:
		return "Engine error";
	case ORT_RUNTIME_EXCEPTION:
		return "Runtime exception";
	case ORT_INVALID_PROTOBUF:
		return "Invalid protobuff";
	case ORT_MODEL_LOADED:
		return "Model loaded";
	case ORT_NOT_IMPLEMENTED:
		return "Not implemented";
	case ORT_INVALID_GRAPH:
		return "Invalid graph";
	case ORT_EP_FAIL:
		return "EP fail";
	case ORT_MODEL_LOAD_CANCELED:
		return "Model load canceled";
	case ORT_MODEL_REQUIRES_COMPILATION:
		return "Model requires compilation";
	default:
		return "Unknown";
	}
}

OnnxStatusException::OnnxStatusException(Ort::Status&& status)
	: std::runtime_error(std::format(
		  "{}: {}",
		  format_ort_error_code(status.GetErrorCode()),
		  status.GetErrorMessage()
	  )),
	  status(std::move(status)) {}

void throw_on_onnx_status(Ort::Status&& status) {
	if (!status.IsOK())
		throw OnnxStatusException(std::move(status));
}

OnnxInvalidInputCount::OnnxInvalidInputCount(size_t actual_count)
	: std::runtime_error(
		  std::format("invalid input count of {}, should be 1", actual_count)
	  ),
	  actual_count(actual_count) {}

OnnxInvalidOutputCount::OnnxInvalidOutputCount(size_t actual_count)
	: std::runtime_error(
		  std::format("invalid output count of {}, should be 1", actual_count)
	  ),
	  actual_count(actual_count) {}