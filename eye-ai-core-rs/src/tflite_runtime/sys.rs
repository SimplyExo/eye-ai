pub type TfLiteRuntimeLogCallback = extern "C" fn(*const std::ffi::c_char);

// see ../../tflite-runtime/include/tflite-runtime/Api.hpp
unsafe extern "C" {
	pub unsafe fn tflite_runtime_create(
		model_data_ptr: *const i8,
		model_data_len: usize,
		delegate_serialization_dir: *const std::ffi::c_char,
		model_token: *const std::ffi::c_char,
		log_warning_callback: TfLiteRuntimeLogCallback,
		log_error_callback: TfLiteRuntimeLogCallback,
		npu_config: u8,
		enable_npu: bool,
		skel_library_dir: *const std::ffi::c_char,
		out_error_msg: *mut *const std::ffi::c_char,
	) -> *mut std::ffi::c_void;
	pub unsafe fn tflite_runtime_free_error_msg(error_msg: *const std::ffi::c_char);
	pub unsafe fn tflite_runtime_run_inference(
		runtime: *mut std::ffi::c_void,
		input_ptr: *mut f32,
		input_len: usize,
		output_ptr: *mut f32,
		output_len: usize,
		out_error_msg: *mut *const std::ffi::c_char,
	);
	pub unsafe fn tflite_runtime_get_input_shape(
		runtime: *mut std::ffi::c_void,
		out_input_shape_ptr: *mut *const i32,
		out_input_shape_len: *mut usize,
	);
	pub unsafe fn tflite_runtime_get_output_shape(
		runtime: *mut std::ffi::c_void,
		out_output_shape_ptr: *mut *const i32,
		out_output_shape_len: *mut usize,
	);
	pub unsafe fn tflite_runtime_destroy(runtime: *mut std::ffi::c_void);
}

#[derive(Debug, Default, Clone, Copy, Hash, PartialEq, Eq)]
pub enum NpuConfigType {
	#[default]
	MiDaS,
	Rel2Abs,
	Yolo,
}
impl NpuConfigType {
	pub fn to_u8_constant(self) -> u8 {
		match self {
			Self::MiDaS => 0,
			Self::Rel2Abs => 1,
			Self::Yolo => 3,
		}
	}
}
