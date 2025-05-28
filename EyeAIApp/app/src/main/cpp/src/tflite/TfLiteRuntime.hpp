#pragma once

#include "TfLiteUtils.hpp"
#include "tflite/c/c_api.h" // IWYU pragma: export
#include "tflite/c/c_api_types.h"
#include "tflite/c/common.h"
#include "tflite_profiler.h"
#include "utils/Profiling.hpp"
#include <cassert>
#include <chrono>
#include <optional>
#include <span>
#include <string>
#include <string_view>

struct TfLiteProfilerEntry {
	std::string name;
	std::chrono::microseconds duration;
};

/** Helper class that wraps the tflite c api */
class TfLiteRuntime {
  private:
	TfLiteModel* model = nullptr;
	TfLiteInterpreter* interpreter = nullptr;
	TfLiteInterpreterOptions* interpreter_options = nullptr;
	/// can be null if GPU delegates are not supported on this device
	TfLiteDelegate* gpu_delegate = nullptr;

	/// data is `this` pointer of TfLiteRuntime
	std::optional<TfLiteTelemetryProfilerStruct> telemetry_profiler =
		std::nullopt;
	std::vector<TfLiteProfilerEntry> current_invoke_profiler_entries;

  public:
	explicit TfLiteRuntime(
		std::span<const int8_t> model_data,
		std::string_view gpu_delegate_serialization_dir,
		std::string_view model_token,
		bool enable_profiling
	);
	~TfLiteRuntime();

	TfLiteRuntime(TfLiteRuntime&&) = delete;
	TfLiteRuntime(const TfLiteRuntime&) = delete;
	void operator=(TfLiteRuntime&&) = delete;
	void operator=(const TfLiteRuntime&) = delete;

	template<typename I, typename O>
	void run_inference(
		std::span<const I> input,
		std::span<O> output,
		std::vector<TfLiteProfilerEntry>& out_profiler_entries
	) {
		PROFILE_DEPTH_FUNCTION()

		load_input<I>(input);
		{
			PROFILE_DEPTH_SCOPE("Invoking of model")

			throw_on_tflite_status(
				TfLiteInterpreterInvoke(interpreter),
				"failed to invoke interpreter"
			);
		}
		read_output<O>(output);

		out_profiler_entries = std::move(current_invoke_profiler_entries);
		current_invoke_profiler_entries.clear();
	}

  private:
	template<typename I>
	void load_input(std::span<const I> input) {
		PROFILE_DEPTH_SCOPE("Loading input")

		auto* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

		if (is_tensor_quantized(input_tensor)) {
			load_quantized_input<I>(input, input_tensor);
		} else {
			load_nonquantized_input(std::as_bytes(input), input_tensor, TFLITE_TYPE_FROM_TYPE<I>);
		}
	}

	template<typename O>
	void read_output(std::span<O> output) {
		PROFILE_DEPTH_SCOPE("Reading output")

		const auto* output_tensor =
			TfLiteInterpreterGetOutputTensor(interpreter, 0);

		if (is_tensor_quantized(output_tensor)) {
			read_quantized_output<O>(output, output_tensor);
		} else {
			read_nonquantized_output(std::as_writable_bytes(output), output_tensor, TFLITE_TYPE_FROM_TYPE<O>);
		}
	}

	static void load_nonquantized_input(
		std::span<const std::byte> input_bytes,
		TfLiteTensor* input_tensor,
		TfLiteType input_type
	);
	static void read_nonquantized_output(
		std::span<std::byte> output_bytes,
		const TfLiteTensor* output_tensor,
		TfLiteType output_type
	);

	template<typename I>
	void
	load_quantized_input(std::span<const I> input, TfLiteTensor* input_tensor) {
		const auto quantized_type_size =
			get_tflite_type_size(input_tensor->type);

		if (!quantized_type_size.has_value())
			throw UnsupportedTypeQuantizationException(input_tensor->type);

		void* quantized_input_data_ptr = TfLiteTensorData(input_tensor);
		if (quantized_input_data_ptr == nullptr)
			throw TensorNotYetCreatedException();
		const auto quantized_input_data_bytes =
			TfLiteTensorByteSize(input_tensor);
		if (quantized_input_data_bytes / *quantized_type_size != input.size())
			throw std::invalid_argument("quantized_input_data_bytes");
		// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
		auto quantized_span = std::span<std::byte>(
			reinterpret_cast<std::byte*>(quantized_input_data_ptr),
			quantized_input_data_bytes
		);
		quantize<I>(
			input, quantized_span, input_tensor->type,
			*reinterpret_cast<const TfLiteAffineQuantization*>(
				input_tensor->quantization.params
			)
		);
		// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
	}

	template<typename O>
	void read_quantized_output(
		std::span<O> output,
		const TfLiteTensor* output_tensor
	) {
		const auto quantized_type_size =
			get_tflite_type_size(output_tensor->type);
		if (!quantized_type_size.has_value())
			throw UnsupportedTypeQuantizationException(output_tensor->type);

		const void* quantized_output_data_ptr = TfLiteTensorData(output_tensor);
		if (quantized_output_data_ptr == nullptr)
			throw TensorNotYetCreatedException();
		const auto quantized_output_data_bytes =
			TfLiteTensorByteSize(output_tensor);
		if (quantized_output_data_bytes / *quantized_type_size != output.size())
			throw std::invalid_argument("quantized_output_data_bytes");
		// NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
		auto quantized_output_span = std::span<const std::byte>(
			reinterpret_cast<const std::byte*>(quantized_output_data_ptr),
			quantized_output_data_bytes
		);

		dequantize<O>(
			quantized_output_span, output, output_tensor->type,
			*reinterpret_cast<const TfLiteAffineQuantization*>(
				output_tensor->quantization.params
			)
		);
		// NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
	}
};