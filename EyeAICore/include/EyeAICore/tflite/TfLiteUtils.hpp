#pragma once

#include "EyeAICore/TensorBuffer.hpp"
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

class ProfilingFrame;

std::string_view format_tflite_type(TfLiteType type);

/// @return byte size of type, or nullopt if type has a dynamic size
std::optional<size_t> get_tflite_type_size(TfLiteType type);

std::string_view format_tflite_status(TfLiteStatus status);

/// @return internal quantization parameters of tensor, or nullopt if
/// tensor is not quantized
[[nodiscard]] std::optional<TfLiteAffineQuantization>
get_tensor_quantization(const TfLiteTensor* tensor);

[[nodiscard]] std::span<const int> get_tensor_shape(const TfLiteTensor* tensor);

/**
 * @param gpu_delegate_serialization_dir Directory where TfLite saves compiled
 * GPU delegate kernels
 * @param model_token unique token to identify the model, should change on model
 * update
 * @param profiling_frame profiling frame used for profiling
 */
[[nodiscard]] std::
	unique_ptr<TfLiteDelegate, decltype(&TfLiteGpuDelegateV2Delete)>
	create_gpu_delegate(
		std::string_view gpu_delegate_serialization_dir,
		std::string_view model_token,
		ProfilingFrame& profiling_frame
	);

enum class NpuConfiguration {
	MiDaS,
	rel2abs,
	Yolo
};

/// @return nullptr if platform does not support qnn delegate right now
[[nodiscard]] std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>


create_qnn_npu_delegate(
	std::string_view delegate_serialization_dir,
	std::string_view model_token,
	NpuConfiguration config
);

/// either a input or a output tensor
class TensorType {
  public:
	enum Type : uint8_t { Input, Output } type;

	TensorType() = delete;
	TensorType(Type type) : type(type) {}

	[[nodiscard]] std::string_view to_string() const;
	bool operator==(const TensorType& other) const = default;
};

struct [[nodiscard]] TfLiteNonFloatTensorTypeError {
	TensorType tensor_type;
	TfLiteType tensor_element_type;

	[[nodiscard]] std::string to_string() const;
	bool operator==(const TfLiteNonFloatTensorTypeError& other) const = default;
};

struct [[nodiscard]] TfLiteTensorsNotCreatedError {
	TensorType tensor_type;

	[[nodiscard]] std::string to_string() const;
	bool operator==(const TfLiteTensorsNotCreatedError& other) const = default;
};

struct [[nodiscard]] TfLiteTensorElementCountMismatch {
	TensorType tensor_type;
	size_t provided_elements;
	size_t expected_elements;

	[[nodiscard]] std::string to_string() const;
	bool
	operator==(const TfLiteTensorElementCountMismatch& other) const = default;
};

struct [[nodiscard]] TfLiteCopyFromInputTensorError {
	TfLiteStatus status;

	[[nodiscard]] std::string to_string() const;
	bool
	operator==(const TfLiteCopyFromInputTensorError& other) const = default;
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
	bool operator==(const InvalidFloat32QuantizationTypeError& other) const =
		default;
};
struct [[nodiscard]] QuantizationElementsMismatch {
	size_t input_elements;
	size_t quantized_out_elements;

	[[nodiscard]] std::string to_string() const;
	bool operator==(const QuantizationElementsMismatch& other) const = default;
};
struct [[nodiscard]] AsymmetricQuantizationError {
	[[nodiscard]] static std::string to_string();
	bool operator==(const AsymmetricQuantizationError& other) const = default;
};
struct [[nodiscard]] InvalidQuantizedType {
	TfLiteType quantized_type;

	[[nodiscard]] std::string to_string() const;
	bool operator==(const InvalidQuantizedType& other) const = default;
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

/// loads input tensor with floats array, supports quantization
[[nodiscard]] std::optional<TfLiteLoadInputError> load_input_tensor_with_floats(
	TfLiteTensor* input_tensor,
	std::span<const float> values,
	ProfilingFrame& profiling_frame
);

struct [[nodiscard]] TfLiteCopyToOutputTensorError {
	TfLiteStatus status;

	[[nodiscard]] std::string to_string() const;
	bool operator==(const TfLiteCopyToOutputTensorError& other) const = default;
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

/// reads floats array from output tensor, supports quantization
[[nodiscard]] std::optional<TfLiteReadOutputError>
read_floats_from_output_tensor(
	const TfLiteTensor* output_tensor,
	std::span<float> output,
	ProfilingFrame& profiling_frame
);

struct [[nodiscard]] TfLiteCreateInterpreterError {
	[[nodiscard]] static std::string to_string();
	bool operator==(const TfLiteCreateInterpreterError& other) const = default;
};

struct [[nodiscard]] TfLiteAllocateTensorsError {
	TfLiteStatus status;

	[[nodiscard]] std::string to_string() const;
	bool operator==(const TfLiteAllocateTensorsError& other) const = default;
};

COMBINED_ERROR(
	TfLiteCreateRuntimeError,
	TfLiteCreateInterpreterError,
	TfLiteAllocateTensorsError
);

struct [[nodiscard]] TfLiteInvokeInterpreterError {
	TfLiteStatus status;

	[[nodiscard]] std::string to_string() const;
	bool operator==(const TfLiteInvokeInterpreterError& other) const = default;
};

struct [[nodiscard]] InvalidInputFormatForModel {
	FloatTensorFormat provided;
	FloatTensorFormat expected;

	[[nodiscard]] std::string to_string() const;
	bool operator==(const InvalidInputFormatForModel& other) const = default;
};

struct [[nodiscard]] InvalidOutputFormatForModel {
	FloatTensorFormat provided;
	FloatTensorFormat expected;

	[[nodiscard]] std::string to_string() const;
	bool operator==(const InvalidOutputFormatForModel& other) const = default;
};

COMBINED_ERROR(
	TfLiteRunInferenceError,
	TfLiteLoadInputError,
	TfLiteInvokeInterpreterError,
	TfLiteReadOutputError,
	InvalidInputFormatForModel,
	InvalidOutputFormatForModel
);