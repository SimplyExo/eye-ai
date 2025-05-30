#pragma once

#include "onnxruntime_cxx_api.h"
#include <stdexcept>
#include <string_view>

std::string_view format_ort_error_code(OrtErrorCode error_code);

class OnnxStatusException : public std::runtime_error {
  public:
	explicit OnnxStatusException(Ort::Status&& status);

	Ort::Status status;
};
void throw_on_onnx_status(Ort::Status&& status);

class OnnxInvalidInputCount : public std::runtime_error {
  public:
	explicit OnnxInvalidInputCount(size_t actual_count);

	size_t actual_count;
};

class OnnxInvalidOutputCount : public std::runtime_error {
  public:
	explicit OnnxInvalidOutputCount(size_t actual_count);

	size_t actual_count;
};