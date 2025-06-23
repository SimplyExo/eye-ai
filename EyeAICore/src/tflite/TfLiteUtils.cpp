#include "EyeAICore/tflite/TfLiteUtils.hpp"

#include "EyeAICore/utils/Errors.hpp"
#include "EyeAICore/utils/Profiling.hpp"

[[nodiscard]] static tl::expected<void, std::string> quantize_floats(
	std::span<const float> values,
	std::span<std::byte> out_quantized_values,
	TfLiteType quantized_type,
	const TfLiteAffineQuantization& quantization
);

[[nodiscard]] static tl::expected<void, std::string> dequantize_to_floats(
	std::span<const std::byte> quantized_values,
	std::span<float> out_values,
	TfLiteType quantized_type,
	const TfLiteAffineQuantization& quantization
);

std::optional<TfLiteAffineQuantization>
get_tensor_quantization(const TfLiteTensor* tensor) {
	if (tensor->quantization.type == kTfLiteNoQuantization)
		return std::nullopt;

	return *static_cast<const TfLiteAffineQuantization*>(
		tensor->quantization.params
	);
}

std::unique_ptr<TfLiteDelegate, decltype(&TfLiteGpuDelegateV2Delete)>
create_gpu_delegate(
	std::string_view gpu_delegate_serialization_dir,
	std::string_view model_token
) {
	PROFILE_DEPTH_FUNCTION()

	TfLiteGpuDelegateOptionsV2 gpu_delegate_options =
		TfLiteGpuDelegateOptionsV2Default();
	gpu_delegate_options.is_precision_loss_allowed = static_cast<int32_t>(true);
	gpu_delegate_options.inference_preference =
		TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER;
	gpu_delegate_options.experimental_flags |=
		TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
	gpu_delegate_options.serialization_dir =
		gpu_delegate_serialization_dir.data();
	gpu_delegate_options.model_token = model_token.data();

	return {
		TfLiteGpuDelegateV2Create(&gpu_delegate_options),
		TfLiteGpuDelegateV2Delete
	};
}

[[nodiscard]] static tl::expected<void, std::string>
load_nonquantized_input_tensor_with_floats(
	TfLiteTensor* input_tensor,
	std::span<const float> values
) {
	PROFILE_DEPTH_FUNCTION()

	if (input_tensor->type != kTfLiteFloat32) {
		return tl::unexpected_fmt(
			"cannot load input tensor with floats, its element type is {}",
			format_tflite_type(input_tensor->type)
		);
	}

	void* tensor_data_ptr = TfLiteTensorData(input_tensor);
	if (tensor_data_ptr == nullptr)
		return tl::unexpected("input tensor not yet created!");

	const auto input_tensor_data_bytes = TfLiteTensorByteSize(input_tensor);
	const auto input_tensor_elements = input_tensor_data_bytes / sizeof(float);
	if (values.size() != input_tensor_elements) {
		return tl::unexpected_fmt(
			"values ({} elements) does not match expected {} "
			"elements from input tensor",
			values.size(), input_tensor_elements
		);
	}
	const TfLiteStatus copy_from_buffer_status = TfLiteTensorCopyFromBuffer(
		input_tensor, values.data(), values.size_bytes()
	);

	if (copy_from_buffer_status == kTfLiteOk)
		return {};

	return tl::unexpected_fmt(
		"failed to copy values to input tensor: {}",
		format_tflite_status(copy_from_buffer_status)
	);
}

[[nodiscard]] static tl::expected<void, std::string>
load_quantized_input_tensor_with_floats(
	TfLiteTensor* input_tensor,
	const TfLiteAffineQuantization& quantization,
	std::span<const float> values
) {
	PROFILE_DEPTH_FUNCTION()

	const auto quantized_type_size = get_tflite_type_size(input_tensor->type);

	if (!quantized_type_size.has_value()) {
		return tl::unexpected_fmt(
			"invalid quantized input type: {}",
			format_tflite_type(input_tensor->type)
		);
	}

	void* quantized_input_data_ptr = TfLiteTensorData(input_tensor);
	if (quantized_input_data_ptr == nullptr)
		return tl::unexpected("quantized input tensor not yet created!");

	const auto quantized_input_data_bytes = TfLiteTensorByteSize(input_tensor);
	const auto quantized_input_elements =
		quantized_input_data_bytes / *quantized_type_size;
	if (values.size() != quantized_input_elements) {
		return tl::unexpected_fmt(
			"values buffer ({} elements) does not match expected {} "
			"elements from tensor",
			values.size(), quantized_input_elements
		);
	}
	const std::span quantized_span(
		static_cast<std::byte*>(quantized_input_data_ptr),
		quantized_input_data_bytes
	);
	return quantize_floats(
		values, quantized_span, input_tensor->type, quantization
	);
}

tl::expected<void, std::string> load_input_tensor_with_floats(
	TfLiteTensor* input_tensor,
	std::span<const float> values
) {
	PROFILE_DEPTH_FUNCTION()

	const auto quantization = get_tensor_quantization(input_tensor);
	if (quantization) {
		return load_quantized_input_tensor_with_floats(
			input_tensor, *quantization, values
		);
	}

	return load_nonquantized_input_tensor_with_floats(input_tensor, values);
}

static tl::expected<void, std::string>
read_floats_from_nonquantized_output_tensor(
	const TfLiteTensor* output_tensor,
	std::span<float> output
) {
	PROFILE_DEPTH_FUNCTION()

	if (output_tensor->type != kTfLiteFloat32) {
		return tl::unexpected_fmt(
			"cannot read floats from output tensor, its element type is {}",
			format_tflite_type(output_tensor->type)
		);
	}

	const auto output_tensor_data_bytes = TfLiteTensorByteSize(output_tensor);
	const auto output_tensor_data_elements =
		output_tensor_data_bytes / sizeof(float);
	if (output.size() != output_tensor_data_elements) {
		return tl::unexpected_fmt(
			"output ({} elements) does not match expected {} "
			"elements from output tensor",
			output.size(), output_tensor_data_elements
		);
	}

	const TfLiteStatus copy_from_buffer_status = TfLiteTensorCopyToBuffer(
		output_tensor, output.data(), output.size_bytes()
	);

	if (copy_from_buffer_status == kTfLiteOk)
		return {};

	return tl::unexpected_fmt(
		"failed to read floats from output tensor: {}",
		format_tflite_status(copy_from_buffer_status)
	);
}

static tl::expected<void, std::string> read_floats_from_quantized_output_tensor(
	const TfLiteTensor* output_tensor,
	std::span<float> output,
	const TfLiteAffineQuantization& quantization
) {
	PROFILE_DEPTH_FUNCTION()

	const auto quantized_type_size = get_tflite_type_size(output_tensor->type);
	if (!quantized_type_size.has_value()) {
		return tl::unexpected_fmt(
			"invalid quantized output type: {}",
			format_tflite_type(output_tensor->type)
		);
	}

	const void* quantized_output_data_ptr = TfLiteTensorData(output_tensor);
	if (quantized_output_data_ptr == nullptr)
		return tl::unexpected("quantized output tensor not yet created!");
	const auto quantized_output_data_bytes =
		TfLiteTensorByteSize(output_tensor);
	const auto quantized_output_elements =
		quantized_output_data_bytes / *quantized_type_size;
	if (quantized_output_elements != output.size()) {
		return tl::unexpected_fmt(
			"output buffer ({} elements) does not match expected {} "
			"elements from tensor",
			output.size(), quantized_output_elements
		);
	}
	const std::span quantized_output_span(
		static_cast<const std::byte*>(quantized_output_data_ptr),
		quantized_output_data_bytes
	);

	return dequantize_to_floats(
		quantized_output_span, output, output_tensor->type, quantization
	);
}

tl::expected<void, std::string> read_floats_from_output_tensor(
	const TfLiteTensor* output_tensor,
	std::span<float> output
) {
	PROFILE_DEPTH_FUNCTION()

	const auto quantization = get_tensor_quantization(output_tensor);
	if (quantization) {
		return read_floats_from_quantized_output_tensor(
			output_tensor, output, *quantization
		);
	}

	return read_floats_from_nonquantized_output_tensor(output_tensor, output);
}

tl::expected<void, std::string> quantize_floats(
	std::span<const float> values,
	std::span<std::byte> out_quantized_values,
	TfLiteType quantized_type,
	const TfLiteAffineQuantization& quantization
) {
	PROFILE_DEPTH_FUNCTION()

	if (quantized_type != kTfLiteUInt8) {
		return tl::unexpected_fmt(
			"unsupported quantization of float32 to {}",
			format_tflite_type(quantized_type)
		);
	}

	if (values.size() != out_quantized_values.size()) {
		return tl::unexpected_fmt(
			"values ({}) and out_quantized_values ({}) dont match",
			values.size(), out_quantized_values.size()
		);
	}

	// for now, only 1 input, 1 output
	if (quantization.scale->size != 1)
		return tl::unexpected("only symmetric quantization supported for now");
	const float quantization_scale = quantization.scale->data[0];
	if (quantization.zero_point->size != 1)
		return tl::unexpected("only symmetric quantization supported for now");
	const int quantization_zero_point = quantization.zero_point->data[0];

	for (size_t i = 0; i < values.size(); i++) {
		static_assert(sizeof(std::byte) == sizeof(uint8_t));
		out_quantized_values[i] = static_cast<std::byte>(
			static_cast<uint8_t>(values[i] / quantization_scale) +
			quantization_zero_point
		);
	}

	return {};
}

tl::expected<void, std::string> dequantize_to_floats(
	std::span<const std::byte> quantized_values,
	std::span<float> out_values,
	TfLiteType quantized_type,
	const TfLiteAffineQuantization& quantization
) {
	PROFILE_DEPTH_FUNCTION()

	if (quantized_type != kTfLiteUInt8) {
		return tl::unexpected_fmt(
			"unsupported dequantization of {} to float32",
			format_tflite_type(quantized_type)
		);
	}

	if (quantized_values.size() != out_values.size()) {
		return tl::unexpected_fmt(
			"out_values ({}) and quantized_values ({}) dont match",
			out_values.size(), quantized_values.size()
		);
	}

	// for now, only 1 input, 1 output
	if (quantization.scale->size != 1)
		return tl::unexpected("only symmetric quantization supported for now");
	const float quantization_scale = quantization.scale->data[0];
	if (quantization.zero_point->size != 1)
		return tl::unexpected("only symmetric quantization supported for now");
	const int quantization_zero_point = quantization.zero_point->data[0];

	for (size_t i = 0; i < out_values.size(); i++) {
		static_assert(sizeof(std::byte) == sizeof(uint8_t));
		const auto quantized = static_cast<const uint8_t>(quantized_values[i]);
		out_values[i] = quantization_scale *
						static_cast<float>(quantized - quantization_zero_point);
	}

	return {};
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