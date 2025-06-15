#pragma once

#include "onnxruntime_cxx_api.h"
#include <string_view>

std::string_view format_ort_error_code(OrtErrorCode error_code);

std::string_view format_ort_element_type(ONNXTensorElementDataType type);