#pragma once

#include "EyeAICore/utils/Errors.hpp"
#include "OnnxUtils.hpp"
#include <onnxruntime_cxx_api.h>
#include <span>

struct OnnxLogCallbacks {
	std::function<void(std::string)> log_info;
	std::function<void(std::string)> log_error;
};

class OnnxRuntime {
  public:
	[[nodiscard]] static tl::expected<std::unique_ptr<OnnxRuntime>, std::string>
	create(
		std::span<const std::byte> model_data,
		OnnxLogCallbacks&& log_callbacks
	);

	OnnxRuntime(OnnxRuntime&&) = default;
	OnnxRuntime(const OnnxRuntime&) = delete;
	OnnxRuntime& operator=(OnnxRuntime&&) = default;
	OnnxRuntime& operator=(const OnnxRuntime&) = delete;
	~OnnxRuntime() = default;

	template<typename I, typename O>
	[[nodiscard]] tl::expected<void, std::string>
	run_inference(std::span<I> input_data, std::span<O> output_data) {
		if (input_type != Ort::TypeToTensorType<I>::type) {
			return tl::unexpected_fmt(
				"input type is {}, but {} was expected",
				format_ort_element_type(Ort::TypeToTensorType<I>::type),
				format_ort_element_type(input_type)
			);
		}
		if (output_type != Ort::TypeToTensorType<O>::type) {
			return tl::unexpected_fmt(
				"output type is {}, but {} was expected",
				format_ort_element_type(Ort::TypeToTensorType<O>::type),
				format_ort_element_type(output_type)
			);
		}

		run_inference_raw(
			std::as_writable_bytes(input_data),
			std::as_writable_bytes(output_data)
		);

		return {};
	}

  private:
	explicit OnnxRuntime(OnnxLogCallbacks&& log_callbacks)
		: log_callbacks(std::move(log_callbacks)) {}

	void run_inference_raw(
		std::span<std::byte> input_data,
		std::span<std::byte> output_data
	);

	Ort::Env env = nullptr;
	Ort::Session session{nullptr};
	Ort::MemoryInfo memory_info{nullptr};
	OnnxLogCallbacks log_callbacks;

	std::string input_name;
	std::vector<int64_t> input_shape;
	ONNXTensorElementDataType input_type =
		ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;

	std::string output_name;
	std::vector<int64_t> output_shape;
	ONNXTensorElementDataType output_type =
		ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
};