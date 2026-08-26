#pragma once

#include <memory>
#include <optional>
#include <span>
#include <string_view>
#if TFLITE_RUNTIME_USE_PREBUILT_TFLITE
#include <tflite/c/c_api.h>
#include <tflite/delegates/gpu/delegate.h>
#else
#include <tensorflow/lite/c/c_api.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>
#endif

using ErrorMsg = std::string;

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
 * @param delegate_serialization_dir Directory where TfLite saves compiled
 * GPU delegate kernels
 * @param model_token unique token to identify the model, should change on model
 * update
 * @param profiling_frame profiling frame used for profiling
 */
[[nodiscard]] std::
	unique_ptr<TfLiteDelegate, decltype(&TfLiteGpuDelegateV2Delete)>
	create_gpu_delegate(
		std::string_view delegate_serialization_dir,
		std::string_view model_token
	);

enum class NpuConfiguration : std::uint8_t { MiDaS, Rel2Abs, Yolo };

/// @return nullptr if platform does not support qnn delegate right now
[[nodiscard]] std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
create_qnn_npu_delegate(
	std::string_view delegate_serialization_dir,
	std::string_view model_token,
	NpuConfiguration config,
	std::string_view skel_library_dir
);

void null_delegate_delete([[maybe_unused]] TfLiteDelegate* delegate);

/// loads input tensor with floats array, supports quantization
[[nodiscard]] std::optional<ErrorMsg> load_input_tensor_with_floats(
	TfLiteTensor* input_tensor,
	std::span<const float> values
);

/// reads floats array from output tensor, supports quantization
[[nodiscard]] std::optional<ErrorMsg> read_floats_from_output_tensor(
	const TfLiteTensor* output_tensor,
	std::span<float> output
);
