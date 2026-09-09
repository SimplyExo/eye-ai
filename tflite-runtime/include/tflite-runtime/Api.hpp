#pragma once

#include <cstddef>
#include <cstdint>

class TfLiteRuntime;
using LogCallback = void (*)(const char*);

extern "C" {
/// `model_data_ptr` needs to live for the lifetime of the TfLiteRuntime
/// if returned pointer is `null`, then `out_error_msg` will have been set to a
/// newly allocated c string. that string then needs to be freed using
/// `tflite_runtime_free_error_msg` eventually
TfLiteRuntime* tflite_runtime_create(
	const int8_t* model_data_ptr,
	size_t model_data_len,
	const char* delegate_serialization_dir,
	const char* model_token,
	LogCallback log_warning_callback,
	LogCallback log_error_callback,
	/// see `NpuConfiguration`
	uint8_t npu_config,
	bool enable_npu,
	const char* skel_library_dir,
	const char** out_error_msg
);
void tflite_runtime_free_error_msg(const char* error_msg);
/// if an error occured, `out_error_msg` will have been set to a newly allocated
/// c string that needs to be freed using `tflite_runtime_free_error_msg`
void tflite_runtime_run_inference(
	TfLiteRuntime* runtime,
	float* input_ptr,
	size_t input_len,
	float* output_ptr,
	size_t output_len,
	const char** out_error_msg
);
/// `out_input_shape_ptr` has lifetime of this TfLiteRuntime
void tflite_runtime_get_input_shape(
	TfLiteRuntime* runtime,
	const int** out_input_shape_ptr,
	size_t* out_input_shape_len
);
/// `out_output_shape_ptr` has lifetime of this TfLiteRuntime
void tflite_runtime_get_output_shape(
	TfLiteRuntime* runtime,
	const int** out_output_shape_ptr,
	size_t* out_output_shape_len
);
void tflite_runtime_destroy(TfLiteRuntime* runtime);
}
