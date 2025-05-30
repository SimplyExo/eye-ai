#pragma once

#include <onnxruntime_cxx_api.h>
#include <span>

struct OnnxLogCallbacks {
	std::function<void(std::string)> log_info;
	std::function<void(std::string)> log_error;
};

class OnnxRuntime {
  public:
	explicit OnnxRuntime(
		std::span<const std::byte> model_data,
		OnnxLogCallbacks&& log_callbacks
	);

	OnnxRuntime(OnnxRuntime&&) = delete;
	OnnxRuntime(const OnnxRuntime&) = delete;
	void operator=(OnnxRuntime&&) = delete;
	void operator=(const OnnxRuntime&) = delete;
	~OnnxRuntime() = default;

	template<typename I, typename O>
	void run_inference(std::span<I> input_data, std::span<O> output_data) {
		if (input_type != Ort::TypeToTensorType<I>::type)
			throw std::invalid_argument("input_type");
		if (output_type != Ort::TypeToTensorType<O>::type)
			throw std::invalid_argument("output_type");

		return run_inference_raw(
			std::as_writable_bytes(input_data),
			std::as_writable_bytes(output_data)
		);
	}

  private:
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
	ONNXTensorElementDataType input_type;

	std::string output_name;
	std::vector<int64_t> output_shape;
	ONNXTensorElementDataType output_type;
};