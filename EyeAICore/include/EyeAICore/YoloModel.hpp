#pragma once

#include "EyeAICore/Operators.hpp"
#include "EyeAICore/tflite/TfLiteRuntime.hpp"

class YoloModel
{
    public:
    YoloModel();

    // Erstellt das Modell
    tl::expected<bool, std::string> create(
		std::vector<int8_t>&& model_data,
		std::string_view gpu_delegate_serialization_dir,
		std::string_view model_token,
		TfLiteLogWarningCallback log_warning_callback,
		TfLiteLogErrorCallback log_error_callback
	);

	tl::expected<void, std::string>
	    run(std::span<float> input, std::span<float> output);

    private:
        std::unique_ptr<TfLiteRuntime> runtime;
};
