#pragma once

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
#include <tl/expected.hpp>

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

[[nodiscard]] tl::expected<void, std::string> load_input_tensor_with_floats(
	TfLiteTensor* input_tensor,
	std::span<const float> values
);

[[nodiscard]] tl::expected<void, std::string> read_floats_from_output_tensor(
	const TfLiteTensor* output_tensor,
	std::span<float> output
);