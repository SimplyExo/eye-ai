#include "EyeAICore/tflite/TfLiteUtils.hpp"
#include "EyeAICore/utils/Profiling.hpp"

TfLiteStatusException::TfLiteStatusException(
	TfLiteStatus status,
	std::string_view context
)
	: std::runtime_error(
		  std::format("{}: {}", context, format_tflite_status(status))
	  ),
	  context(context), status(status) {}

void throw_on_tflite_status(TfLiteStatus status, std::string_view context) {
	if (status != kTfLiteOk)
		throw TfLiteStatusException(status, context);
}

UnsupportedTypeQuantizationException::UnsupportedTypeQuantizationException(
	TfLiteType unsupported_type
)
	: std::runtime_error(
		  std::format(
			  "unsupported quantization type: {}",
			  format_tflite_type(unsupported_type)
		  )
	  ) {}

WrongTypeException::WrongTypeException(
	TfLiteType expected_type,
	TfLiteType provided_type
)
	: std::runtime_error(
		  std::format(
			  "invalid type of {}, expected {}",
			  format_tflite_type(provided_type),
			  format_tflite_type(expected_type)
		  )
	  ) {}

TfLiteDelegate* create_gpu_delegate(
	std::string_view gpu_delegate_serialization_dir,
	std::string_view model_token
) {
	PROFILE_DEPTH_FUNCTION()

	TfLiteGpuDelegateOptionsV2 gpu_delegate_options =
		TfLiteGpuDelegateOptionsV2Default();
	gpu_delegate_options.is_precision_loss_allowed = (int32_t)true;
	gpu_delegate_options.inference_preference =
		TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
	gpu_delegate_options.experimental_flags |= TfLiteGpuExperimentalFlags::
		TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
	gpu_delegate_options.serialization_dir =
		gpu_delegate_serialization_dir.data();
	gpu_delegate_options.model_token = model_token.data();

	return TfLiteGpuDelegateV2Create(&gpu_delegate_options);
}

void delete_gpu_delegate(TfLiteDelegate* delegate) {
	TfLiteGpuDelegateV2Delete(delegate);
}

template<>
void quantize<float>(
	std::span<const float> values,
	std::span<std::byte> quantized_values,
	TfLiteType quantized_type,
	const TfLiteAffineQuantization& quantization
) {
	PROFILE_DEPTH_FUNCTION()

	if (quantized_type != kTfLiteUInt8)
		throw UnsupportedTypeQuantizationException(quantized_type);

	if (values.size() != quantized_values.size())
		throw std::invalid_argument("values and quantized_values");

	// for now, only 1 input, 1 output
	if (quantization.scale->size != 1)
		throw UnsupportedAsymmetricQuantizationException();
	const float quantization_scale = quantization.scale->data[0];
	if (quantization.zero_point->size != 1)
		throw UnsupportedAsymmetricQuantizationException();
	const int quantization_zero_point = quantization.zero_point->data[0];

	for (size_t i = 0; i < values.size(); i++) {
		static_assert(sizeof(std::byte) == sizeof(uint8_t));
		quantized_values[i] =
			(std::byte)((uint8_t)(values[i] / quantization_scale) +
						quantization_zero_point);
	}
}

template<>
void dequantize<float>(
	std::span<const std::byte> quantized_values,
	std::span<float> real_values,
	TfLiteType quantized_type,
	const TfLiteAffineQuantization& quantization
) {
	PROFILE_DEPTH_FUNCTION()

	if (quantized_type != kTfLiteUInt8)
		throw UnsupportedTypeQuantizationException(quantized_type);

	if (quantized_values.size() != real_values.size())
		throw std::invalid_argument("real_values and quantized_values");

	// for now, only 1 input, 1 output
	if (quantization.scale->size != 1)
		throw UnsupportedAsymmetricQuantizationException();
	const float quantization_scale = quantization.scale->data[0];
	if (quantization.zero_point->size != 1)
		throw UnsupportedAsymmetricQuantizationException();
	const int quantization_zero_point = quantization.zero_point->data[0];

	for (size_t i = 0; i < real_values.size(); i++) {
		static_assert(sizeof(std::byte) == sizeof(uint8_t));
		const auto quantized = (const uint8_t)quantized_values[i];
		real_values[i] =
			quantization_scale * (float)(quantized - quantization_zero_point);
	}
}

std::optional<size_t> get_tflite_type_size(TfLiteType type) {
	switch (type) {
	default:
		return std::nullopt;
	case kTfLiteFloat32:
		return sizeof(float);
	case kTfLiteInt32:
		return sizeof(int32_t);
	case kTfLiteUInt8:
		return sizeof(uint8_t);
	case kTfLiteInt64:
		return sizeof(int64_t);
	case kTfLiteBool:
		return sizeof(bool);
	case kTfLiteInt16:
		return sizeof(int16_t);
	case kTfLiteInt8:
		return sizeof(int8_t);
	case kTfLiteFloat16:
		return 2;
	case kTfLiteFloat64:
		return sizeof(double);
	case kTfLiteUInt64:
		return sizeof(uint64_t);
	case kTfLiteUInt32:
		return sizeof(uint32_t);
	case kTfLiteUInt16:
		return sizeof(uint16_t);
	case kTfLiteBFloat16:
		return 2;
	}
}

std::string_view format_tflite_type(TfLiteType type) {
	switch (type) {
	default:
		return "unknown";
	case kTfLiteNoType:
		return "no type";
	case kTfLiteFloat32:
		return "float32";
	case kTfLiteInt32:
		return "int32";
	case kTfLiteUInt8:
		return "uint8";
	case kTfLiteInt64:
		return "int64";
	case kTfLiteString:
		return "string";
	case kTfLiteBool:
		return "bool";
	case kTfLiteInt16:
		return "int16";
	case kTfLiteComplex64:
		return "complex64";
	case kTfLiteInt8:
		return "int8";
	case kTfLiteFloat16:
		return "float16";
	case kTfLiteFloat64:
		return "float64";
	case kTfLiteComplex128:
		return "complex128";
	case kTfLiteUInt64:
		return "uint64";
	case kTfLiteResource:
		return "resource";
	case kTfLiteVariant:
		return "variant";
	case kTfLiteUInt32:
		return "uint32";
	case kTfLiteUInt16:
		return "uint16";
	case kTfLiteInt4:
		return "int4";
	case kTfLiteBFloat16:
		return "bfloat16";
	}
}

std::string_view format_tflite_status(TfLiteStatus status) {
	switch (status) {
	case kTfLiteOk:
		return "ok";
	case kTfLiteError:
		return "general error";
	case kTfLiteDelegateError:
		return "delegate error";
	case kTfLiteApplicationError:
		return "application error";
	case kTfLiteDelegateDataNotFound:
		return "delegate data not found";
	case kTfLiteDelegateDataWriteError:
		return "delegate data write error";
	case kTfLiteDelegateDataReadError:
		return "delegate data read error";
	case kTfLiteUnresolvedOps:
		return "unresolved Ops";
	case kTfLiteCancelled:
		return "canceled";
	case kTfLiteOutputShapeNotKnown:
		return "output shape not known";
	default:
		return "unknown";
	}
}