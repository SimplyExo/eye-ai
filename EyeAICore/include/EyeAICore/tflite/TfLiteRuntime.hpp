#pragma once

#include "EyeAICore/utils/Errors.hpp"
#include "EyeAICore/utils/Profiling.hpp"
#include "TfLiteUtils.hpp"
#include "tflite/c/c_api.h" // IWYU pragma: export
#include "tflite/c/c_api_types.h"
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tl/expected.hpp>

using TfLiteLogWarningCallback = void (*)(std::string);
using TfLiteLogErrorCallback = void (*)(std::string);

struct TfLiteErrorReporterUserData {
	TfLiteLogWarningCallback log_warning_callback;
	TfLiteLogErrorCallback log_error_callback;
};

/** Helper class that wraps the tflite c api */
class TfLiteRuntime {
	std::vector<int8_t> model_data;
	std::unique_ptr<TfLiteModel, decltype(&TfLiteModelDelete)> model{
		nullptr, TfLiteModelDelete
	};
	std::unique_ptr<TfLiteInterpreter, decltype(&TfLiteInterpreterDelete)>
		interpreter{nullptr, TfLiteInterpreterDelete};
	std::unique_ptr<
		TfLiteInterpreterOptions,
		decltype(&TfLiteInterpreterOptionsDelete)>
		interpreter_options{nullptr, TfLiteInterpreterOptionsDelete};
	/// can be null if GPU delegates are not supported on this device
	std::unique_ptr<TfLiteDelegate, decltype(&TfLiteGpuDelegateV2Delete)>
		gpu_delegate{nullptr, TfLiteGpuDelegateV2Delete};

	TfLiteErrorReporterUserData error_reporter_user_data;

  public:
	[[nodiscard]] static tl::
		expected<std::unique_ptr<TfLiteRuntime>, std::string>
		create(
			std::vector<int8_t>&& model_data,
			std::string_view gpu_delegate_serialization_dir,
			std::string_view model_token,
			TfLiteLogWarningCallback log_warning_callback,
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

		return load_input<I>(input).and_then([this]() { return invoke(); }
		).and_then([this, output]() { return read_output<O>(output); });
	}

  private:
	explicit TfLiteRuntime(
		std::vector<int8_t>&& model_data,
		TfLiteErrorReporterUserData error_reporter_user_data
	)
		: model_data(std::move(model_data)),
		  error_reporter_user_data(error_reporter_user_data) {}

	[[nodiscard]] tl::expected<void, std::string> invoke();

	template<typename I>
	[[nodiscard]] tl::expected<void, std::string>
	load_input(std::span<const I> input) {
		PROFILE_DEPTH_SCOPE("Loading input")

		auto* input_tensor =
			TfLiteInterpreterGetInputTensor(interpreter.get(), 0);

		return is_tensor_quantized(input_tensor)
				   ? load_quantized_input<I>(input, input_tensor)
				   : load_nonquantized_input(std::as_bytes(input), input_tensor, TFLITE_TYPE_FROM_TYPE<I>);
	}

	template<typename O>
	[[nodiscard]] tl::expected<void, std::string>
	read_output(std::span<O> output) {
		PROFILE_DEPTH_SCOPE("Reading output")

		const auto* output_tensor =
			TfLiteInterpreterGetOutputTensor(interpreter.get(), 0);

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