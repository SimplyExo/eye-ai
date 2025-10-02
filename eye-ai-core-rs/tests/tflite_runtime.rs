use eye_ai_core_rs::{
	FLOAT_TENSOR_BUFFER_MIDAS_IMAGE_RBG_FORMAT, FLOAT_TENSOR_BUFFER_RAW_RELATIVE_DEPTH_FORMAT,
	TensorBuffer,
	tflite::{CreateTfLiteRuntimeInfo, TfLiteRuntime},
};
use std::{
	ffi::{CStr, CString},
	path::PathBuf,
};

#[cfg(target_family = "unix")]
unsafe fn load_library_global<P: AsRef<std::ffi::OsStr>>(
	path: P,
) -> Result<libloading::os::unix::Library, libloading::Error> {
	unsafe {
		libloading::os::unix::Library::open(
			Some(path),
			libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL,
		)
	}
}

#[cfg(not(target_family = "unix"))]
unsafe fn load_library_global<P: AsRef<std::ffi::OsStr>>(
	path: P,
) -> Result<libloading::Library, libloading::Error> {
	unsafe { Library::new(path) }
}

#[test]
fn test_tflite_runtime() {
	let _libabsl_log_internal_nullguard =
		unsafe { load_library_global("libabsl_log_internal_nullguard.so").unwrap() };

	let tmp_dir = std::env::temp_dir().join("eye-ai-core-rs-testing-serialization-cache");
	std::fs::create_dir_all(&tmp_dir)
		.expect("failed to create tmp cache directory for delegate serialization");
	let tmp_dir_str = CString::new(tmp_dir.to_string_lossy().into_owned()).unwrap();

	let model_token = CString::new("midas_model_token").unwrap();

	let tflite_runtime_create_info = CreateTfLiteRuntimeInfo {
		tflite_lib_filepath: PathBuf::from("libtensorflow-lite.so"),
		tflite_gpu_delegate_lib_filepath: None,
		model_data: std::fs::read("../EyeAIApp/app/src/main/assets/midas_v2_1_256x256.tflite")
			.expect("failed to load midas.tflite model file"),
		gpu_delegate_serialization_dir: tmp_dir_str.as_c_str(),
		model_token: model_token.as_c_str(),
		model_input_format: FLOAT_TENSOR_BUFFER_MIDAS_IMAGE_RBG_FORMAT,
		model_output_format: FLOAT_TENSOR_BUFFER_RAW_RELATIVE_DEPTH_FORMAT,
		log_warning_callback: |message| println!("[WARN] {}", message),
		log_error_callback: |message| unsafe {
			match CStr::from_ptr(message).to_str() {
				Ok(msg) => println!("[ERROR] {}", msg),
				Err(_) => println!(
					"[ERROR] Failed to convert CString to String in order to log tflite error message"
				),
			}
		},
		npu_config: None,
	};

	let mut tflite_runtime =
		TfLiteRuntime::new(tflite_runtime_create_info).expect("failed to create interpreter");

	let mut input_tensor_buffer_container = [0.0f32; 256 * 256 * 3];
	let input_tensor_buffer = TensorBuffer::<f32, FLOAT_TENSOR_BUFFER_MIDAS_IMAGE_RBG_FORMAT>::new(
		&mut input_tensor_buffer_container,
	);

	let _output_tensor_buffer: TensorBuffer<f32, FLOAT_TENSOR_BUFFER_RAW_RELATIVE_DEPTH_FORMAT> =
		tflite_runtime
			.run_inference_with_tensors(input_tensor_buffer)
			.expect("failed to run inference on midas model");
}
