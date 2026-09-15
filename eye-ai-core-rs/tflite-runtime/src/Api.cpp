#include "tflite-runtime/Api.hpp"
#include "tflite-runtime/TfLiteRuntime.hpp"
#include <cstring>

inline static const char* allocate_error_msg(const std::string& error_msg) {
	// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
	char* const copied_error_msg = new char[error_msg.size()];
	std::strcpy(copied_error_msg, error_msg.c_str());
	return copied_error_msg;
}

void tflite_runtime_free_error_msg(const char* error_msg) {
	delete[] error_msg; // NOLINT(cppcoreguidelines-owning-memory)
}

TfLiteRuntime* tflite_runtime_create(
	const int8_t* model_data_ptr,
	size_t model_data_len,
	const char* delegate_serialization_dir,
	const char* model_token,
	LogCallback log_warning_callback,
	LogCallback log_error_callback,
	// NpuConfiguration
	uint8_t npu_config,
	bool enable_npu,
	const char* skel_library_dir,
	const char** out_error_msg
) {
	const std::span<const int8_t> model_data(model_data_ptr, model_data_len);
	auto result = TfLiteRuntime::create(
		model_data, delegate_serialization_dir, model_token,
		log_warning_callback, log_error_callback,
		static_cast<NpuConfiguration>(npu_config), enable_npu, skel_library_dir
	);
	if (result.has_value()) {
		*out_error_msg = nullptr;
		return result.value().release();
	}

	*out_error_msg = allocate_error_msg(result.error());
	return nullptr;
}

void tflite_runtime_run_inference(
	TfLiteRuntime* runtime,
	float* input_ptr,
	size_t input_len,
	float* output_ptr,
	size_t output_len,
	const char** out_error_msg
) {
	const std::span<float> input(input_ptr, input_len);
	const std::span<float> output(output_ptr, output_len);
	auto error = runtime->run_inference(input, output);
	if (error)
		*out_error_msg = allocate_error_msg(error.value());
	else
		*out_error_msg = nullptr;
}

void tflite_runtime_get_input_shape(
	TfLiteRuntime* runtime,
	const int** out_input_shape_ptr,
	size_t* out_input_shape_len
) {
	const std::span<const int> input_shape = runtime->get_input_shape();
	*out_input_shape_ptr = input_shape.data();
	*out_input_shape_len = input_shape.size();
}

void tflite_runtime_get_output_shape(
	TfLiteRuntime* runtime,
	const int** out_output_shape_ptr,
	size_t* out_output_shape_len
) {
	const std::span<const int> output_shape = runtime->get_output_shape();
	*out_output_shape_ptr = output_shape.data();
	*out_output_shape_len = output_shape.size();
}

void tflite_runtime_destroy(TfLiteRuntime* runtime) {
	runtime->~TfLiteRuntime();
}
