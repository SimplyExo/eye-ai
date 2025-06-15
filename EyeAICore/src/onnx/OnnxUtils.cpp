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

std::string_view format_ort_element_type(ONNXTensorElementDataType type) {
	switch (type) {
	default:
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
		return "undefined";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
		return "float";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
		return "uint8";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
		return "int8";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
		return "uint16";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
		return "int16";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
		return "int32";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
		return "int64";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
		return "string";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
		return "bool";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
		return "float16";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
		return "double";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
		return "uint32";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
		return "uint64";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
		return "complex64";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
		return "complex128";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
		return "bfloat16";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN:
		return "float8e4m3fn";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ:
		return "float8e4m3fnuz";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2:
		return "float8e5m2";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ:
		return "float8e5m2fnuz";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4:
		return "uint4";
	case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
		return "int4";
	}
}