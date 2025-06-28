#pragma once

#include "EyeAICore/Operators.hpp"
#include "EyeAICore/utils/Errors.hpp"
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#if EYE_AI_CORE_USE_PREBUILT_TFLITE
#include <tflite/c/c_api.h>
#include <tflite/delegates/gpu/delegate.h>
#else
#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>
#endif

std::string_view format_tflite_type(TfLiteType type);

std::optional<size_t> get_tflite_type_size(TfLiteType type);

std::string_view format_tflite_status(TfLiteStatus status);

[[nodiscard]] static std::optional<TfLiteAffineQuantization>
get_tensor_quantization(const TfLiteTensor* tensor);

[[nodiscard]] std::
	unique_ptr<TfLiteDelegate, decltype(&TfLiteGpuDelegateV2Delete)>
	create_gpu_delegate(
		std::string_view gpu_delegate_serialization_dir,
		std::string_view model_token
	);

class TensorType {
  public:
	enum Type : uint8_t { Input, Output } type;

	TensorType() = delete;
	TensorType(Type type) : type(type) {}

	[[nodiscard]] std::string_view to_string() const;
};

struct [[nodiscard]] TfLiteNonFloatTensorTypeError {
	TensorType tensor_type;
	TfLiteType tensor_element_type;

	[[nodiscard]] std::string to_string() const;
};

struct [[nodiscard]] TfLiteTensorsNotCreatedError {
	TensorType tensor_type;

	[[nodiscard]] std::string to_string() const;
};

struct [[nodiscard]] TfLiteTensorElementCountMismatch {
	TensorType tensor_type;
	size_t provided_elements;
	size_t expected_elements;

	[[nodiscard]] std::string to_string() const;
};

struct [[nodiscard]] TfLiteCopyFromInputTensorError {
	TfLiteStatus status;

	[[nodiscard]] std::string to_string() const;
};

COMBINED_ERROR(
	TfLiteLoadNonQuantizedInputError,
	TfLiteTensorsNotCreatedError,
	TfLiteNonFloatTensorTypeError,
	TfLiteTensorElementCountMismatch,
	TfLiteCopyFromInputTensorError
);
struct [[nodiscard]] InvalidFloat32QuantizationTypeError {
	TfLiteType quantized_type;

	[[nodiscard]] std::string to_string() const;
};
struct [[nodiscard]] QuantizationElementsMismatch {
	size_t input_elements;
	size_t quantized_out_elements;

	[[nodiscard]] std::string to_string() const;
};
struct [[nodiscard]] AsymmetricQuantizationError {
	[[nodiscard]] static std::string to_string();
};
struct [[nodiscard]] InvalidQuantizedType {
	TfLiteType quantized_type;

	[[nodiscard]] std::string to_string() const;
};
COMBINED_ERROR(
	QuantizeFloatError,
	InvalidFloat32QuantizationTypeError,
	QuantizationElementsMismatch,
	AsymmetricQuantizationError
);
COMBINED_ERROR(
	TfLiteLoadQuantizedInputError,
	TfLiteTensorsNotCreatedError,
	TfLiteTensorElementCountMismatch,
	InvalidQuantizedType,
	QuantizeFloatError
);
COMBINED_ERROR(
	TfLiteLoadInputError,
	TfLiteLoadNonQuantizedInputError,
	TfLiteLoadQuantizedInputError
);

[[nodiscard]] std::optional<TfLiteLoadInputError> load_input_tensor_with_floats(
	TfLiteTensor* input_tensor,
	std::span<const float> values
);

struct [[nodiscard]] TfLiteCopyToOutputTensorError {
	TfLiteStatus status;

	[[nodiscard]] std::string to_string() const;
};

COMBINED_ERROR(
	TfLiteReadNonQuantizedOutputError,
	TfLiteNonFloatTensorTypeError,
	TfLiteTensorElementCountMismatch,
	TfLiteCopyToOutputTensorError
);
COMBINED_ERROR(
	DequantizeFloatError,
	InvalidFloat32QuantizationTypeError,
	QuantizationElementsMismatch,
	AsymmetricQuantizationError
);
COMBINED_ERROR(
	TfLiteReadQuantizedOutputError,
	TfLiteTensorsNotCreatedError,
	TfLiteTensorElementCountMismatch,
	DequantizeFloatError
);
COMBINED_ERROR(
	TfLiteReadOutputError,
	TfLiteReadNonQuantizedOutputError,
	TfLiteReadQuantizedOutputError
);

[[nodiscard]] std::optional<TfLiteReadOutputError>
read_floats_from_output_tensor(
	const TfLiteTensor* output_tensor,
	std::span<float> output
);

struct [[nodiscard]] TfLiteCreateInterpreterError {
	[[nodiscard]] std::string to_string() const;
};

struct [[nodiscard]] TfLiteAllocateTensorsError {
	TfLiteStatus status;

	[[nodiscard]] std::string to_string() const;
};

COMBINED_ERROR(
	TfLiteCreateRuntimeError,
	TfLiteCreateInterpreterError,
	TfLiteAllocateTensorsError
);

struct [[nodiscard]] TfLiteInvokeInterpreterError {
	TfLiteStatus status;

	[[nodiscard]] std::string to_string() const;
};

COMBINED_ERROR(
	TfLiteRunInferenceError,
	OperatorError,
	TfLiteLoadInputError,
	TfLiteInvokeInterpreterError,
	TfLiteReadOutputError
);