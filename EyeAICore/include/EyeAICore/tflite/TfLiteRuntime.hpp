#pragma once

#include "EyeAICore/utils/Errors.hpp"
#include "EyeAICore/utils/Profiling.hpp"
#include "TfLiteUtils.hpp"
#include "tflite/c/c_api.h" // IWYU pragma: export
#include "tflite/c/c_api_types.h"
#include <chrono>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tl/expected.hpp>

struct TfLiteProfilerEntry {
	std::string name;
	std::chrono::microseconds duration;
};

using TfLiteLogErrorCallback = void (*)(std::string);

/** Helper class that wraps the tflite c api */
class TfLiteRuntime {
	TfLiteModel* model = nullptr;
	TfLiteInterpreter* interpreter = nullptr;
	TfLiteInterpreterOptions* interpreter_options = nullptr;
	/// can be null if GPU delegates are not supported on this device
	TfLiteDelegate* gpu_delegate = nullptr;

	TfLiteLogErrorCallback log_error_callback;

  public:
	[[nodiscard]] static tl::
		expected<std::unique_ptr<TfLiteRuntime>, std::string>
		create(
			std::span<const int8_t> model_data,
			std::string_view gpu_delegate_serialization_dir,
			std::string_view model_token,
			TfLiteLogErrorCallback log_error_callback
		);

	~TfLiteRuntime();

	TfLiteRuntime(TfLiteRuntime&&) = delete;
	TfLiteRuntime(const TfLiteRuntime&) = delete;
	void operator=(TfLiteRuntime&&) = delete;
	void operator=(const TfLiteRuntime&) = delete;

	template<typename I, typename O>
	[[nodiscard]] tl::expected<void, std::string>
	run_inference(std::span<const I> input, std::span<O> output) {
		PROFILE_DEPTH_FUNCTION()

		return load_input<I>(input).and_then([&]() { return invoke(); }
		).and_then([&]() { return read_output<O>(output); });
	}

  private:
	explicit TfLiteRuntime(TfLiteLogErrorCallback log_error_callback)
		: log_error_callback(log_error_callback) {}

	[[nodiscard]] tl::expected<void, std::string> invoke();

	template<typename I>
	[[nodiscard]] tl::expected<void, std::string>
	load_input(std::span<const I> input) {
		PROFILE_DEPTH_SCOPE("Loading input")

		auto* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

		return is_tensor_quantized(input_tensor)
				   ? load_quantized_input<I>(input, input_tensor)
				   : load_nonquantized_input(std::as_bytes(input), input_tensor, TFLITE_TYPE_FROM_TYPE<I>);
	}

	template<typename O>
	[[nodiscard]] tl::expected<void, std::string>
	read_output(std::span<O> output) {
		PROFILE_DEPTH_SCOPE("Reading output")

		const auto* output_tensor =
			TfLiteInterpreterGetOutputTensor(interpreter, 0);

		return is_tensor_quantized(output_tensor)
				   ? read_quantized_output<O>(output, output_tensor)
				   : read_nonquantized_output(std::as_writable_bytes(output), output_tensor, TFLITE_TYPE_FROM_TYPE<O>);
	}

	[[nodiscard]] static tl::expected<void, std::string>
	load_nonquantized_input(
		std::span<const std::byte> input_bytes,
		TfLiteTensor* input_tensor,
		TfLiteType input_type
	);
	[[nodiscard]] static tl::expected<void, std::string>
	read_nonquantized_output(
		std::span<std::byte> output_bytes,
		const TfLiteTensor* output_tensor,
		TfLiteType output_type
	);

	template<typename I>
	[[nodiscard]] tl::expected<void, std::string>
	load_quantized_input(std::span<const I> input, TfLiteTensor* input_tensor) {
		const auto quantized_type_size =
			get_tflite_type_size(input_tensor->type);

		if (!quantized_type_size.has_value()) {
			return tl::unexpected_fmt(
				"invalid qunatized input type: {}",
				format_tflite_type(input_tensor->type)
			);
		}

		void* quantized_input_data_ptr = TfLiteTensorData(input_tensor);
		if (quantized_input_data_ptr == nullptr)
			return tl::unexpected("quantized input tensor not yet created!");

		const auto quantized_input_data_bytes =
			TfLiteTensorByteSize(input_tensor);
		const auto quantized_input_elements =
			quantized_input_data_bytes / *quantized_type_size;
		if (input.size() != quantized_input_elements) {
			return tl::unexpected_fmt(
				"input buffer ({} elements) does not match expected {} "
				"elements from tensor",
				input.size(), quantized_input_elements
			);
		}
		const std::span quantized_span(
			static_cast<std::byte*>(quantized_input_data_ptr),
			quantized_input_data_bytes
		);
		return quantize<I>(
			input, quantized_span, input_tensor->type,
			*static_cast<const TfLiteAffineQuantization*>(
				input_tensor->quantization.params
			)
		);
	}

	template<typename O>
	[[nodiscard]] tl::expected<void, std::string> read_quantized_output(
		std::span<O> output,
		const TfLiteTensor* output_tensor
	) {
		const auto quantized_type_size =
			get_tflite_type_size(output_tensor->type);
		if (!quantized_type_size.has_value()) {
			return tl::unexpected_fmt(
				"invalid qunatized output type: {}",
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

		return dequantize<O>(
			quantized_output_span, output, output_tensor->type,
			*static_cast<const TfLiteAffineQuantization*>(
				output_tensor->quantization.params
			)
		);
	}
};