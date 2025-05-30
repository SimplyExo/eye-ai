#pragma once

#include <cassert>
#include <span>
#include <string_view>
#include <stdexcept>
#include <optional>
#include <tflite/c/c_api.h>
#include <tflite/c/common.h>
#include <tflite/delegates/gpu/delegate.h>

std::string_view format_tflite_type(TfLiteType type);

std::string_view format_tflite_status(TfLiteStatus status);

inline static bool is_tensor_quantized(const TfLiteTensor* tensor) {
	return tensor->quantization.type == kTfLiteAffineQuantization;
}

class TfLiteStatusException : public std::runtime_error {
  public:
	explicit TfLiteStatusException(
		TfLiteStatus status,
		std::string_view context
	);

	std::string_view context;
	TfLiteStatus status;
};
void throw_on_tflite_status(TfLiteStatus status, std::string_view context);

class UnsupportedTypeQuantizationException : public std::runtime_error {
  public:
	explicit UnsupportedTypeQuantizationException(TfLiteType unsupported_type);
};

class UnsupportedAsymmetricQuantizationException : public std::exception {
  public:
	[[nodiscard]] const char* what() const noexcept override {
		return "asymmetric quantization unsupported";
	}
};

class TensorNotYetCreatedException : public std::exception {
  public:
	[[nodiscard]] const char* what() const noexcept override {
		return "tensor not yet created";
	}
};

class WrongTypeException : public std::runtime_error {
  public:
	explicit WrongTypeException(
		TfLiteType expected_type,
		TfLiteType provided_type
	);
};

template<typename T>
void quantize(
	std::span<const T> values,
	std::span<std::byte> quantized_values,
	TfLiteType quantized_type,
	const TfLiteAffineQuantization& quantization
);

template<typename T>
void dequantize(
	std::span<const std::byte> quantized_values,
	std::span<T> real_values,
	TfLiteType quantized_type,
	const TfLiteAffineQuantization& quantization
);

TfLiteDelegate* create_gpu_delegate(
	std::string_view gpu_delegate_serialization_dir,
	std::string_view model_token
);
void delete_gpu_delegate(TfLiteDelegate* delegate);

std::optional<size_t> get_tflite_type_size(TfLiteType type);

template<typename T>
inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE = kTfLiteNoType;
// clang-format off
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<float> = kTfLiteFloat32;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<int32_t> = kTfLiteInt32;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<uint8_t> = kTfLiteUInt8;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<int64_t> = kTfLiteInt64;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<bool> = kTfLiteBool;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<int16_t> = kTfLiteInt16;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<int8_t> = kTfLiteInt8;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<TfLiteFloat16> = kTfLiteFloat16;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<double> = kTfLiteFloat64;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<uint64_t> = kTfLiteUInt64;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<uint32_t> = kTfLiteUInt32;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<uint16_t> = kTfLiteUInt16;
template<> inline constexpr TfLiteType TFLITE_TYPE_FROM_TYPE<TfLiteBFloat16> = kTfLiteBFloat16;
// clang-format on
