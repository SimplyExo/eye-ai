#pragma once

#include "EyeAICore/TensorBuffer.hpp"
#include "EyeAICore/tflite/TfLiteRuntime.hpp"

class NLPModel {
  public:
	NLPModel() = default;

	struct ClassResults {
		float TEXT_RECOGNITION = 0;
		float OBJECT_DETECTION = 0;
		float CHANGE_SPEECH_SPEED = 0;
		float CHANGE_SPEAKER = 0;
		float REDIRECT_TO_LLM = 0;
		float OPEN_SETTINGS = 0;
		float ABORT = 0;
		float NOISE = 0;
	};

	// Erstellt das Modell
	tl::expected<bool, std::string> create(
		std::vector<int8_t>&& model_data,
		std::string_view gpu_delegate_serialization_dir,
		std::string_view model_token,
		TfLiteLogWarningCallback log_warning_callback,
		TfLiteLogErrorCallback log_error_callback
	);

	tl::expected<std::vector<ClassResults>, std::string>
	run(FloatTensorBuffer<FloatTensorFormat::NLPOutput>& input);

	std::span<const int> get_input_shape();

	std::span<const int> get_output_shape();

	size_t num_channel = 0;
	size_t num_elements = 0;

  private:
	std::unique_ptr<TfLiteRuntime> runtime;
};
