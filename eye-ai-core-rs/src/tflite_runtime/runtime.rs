use std::ffi::NulError;
use std::ffi::{CStr, CString};

use thiserror::Error;
use tracing::{error, warn};

use crate::tflite_runtime::NpuConfig;
use crate::tflite_runtime::sys;
use crate::{FloatTensorBuffer, FloatTensorFormat, check_float_tensor_format};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TfLiteError(String);
impl From<CString> for TfLiteError {
	fn from(value: CString) -> Self {
		Self(
			value
				.to_str()
				.expect("invalid tflite cstring error msg")
				.to_string(),
		)
	}
}
impl std::error::Error for TfLiteError {
	fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
		None
	}
	fn description(&self) -> &str {
		self.0.as_str()
	}
	fn cause(&self) -> Option<&dyn std::error::Error> {
		None
	}
}
impl std::fmt::Display for TfLiteError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self.0)
	}
}

#[derive(Debug, Clone)]
pub struct CreateTfLiteRuntimeInfo {
	pub model_data: Vec<u8>,
	pub delegate_serialization_dir: String,
	pub model_token: String,
	pub model_input_format: FloatTensorFormat,
	pub model_output_format: FloatTensorFormat,
	pub npu_config: Option<NpuConfig>,
}

#[derive(Debug, Clone, Error)]
pub enum CreateTfLiteRuntimeError {
	#[error("{0}")]
	TfLiteError(#[from] TfLiteError),
	#[error("{0}")]
	NulError(#[from] NulError),
}

// (unused fields are here for lifetime guaranties)
#[derive(Debug)]
pub struct TfLiteRuntime {
	runtime: *mut std::ffi::c_void,
	#[allow(unused)]
	delegate_serialization_dir: CString,
	#[allow(unused)]
	model_data: Vec<u8>,
	#[allow(unused)]
	model_token: CString,
	model_input_format: FloatTensorFormat,
	model_output_format: FloatTensorFormat,
}
unsafe impl Send for TfLiteRuntime {}
unsafe impl Sync for TfLiteRuntime {}
impl TfLiteRuntime {
	pub fn new(create_info: CreateTfLiteRuntimeInfo) -> Result<Self, CreateTfLiteRuntimeError> {
		let delegate_serialization_dir_cstr = CString::new(create_info.delegate_serialization_dir)?;
		let model_token_cstr = CString::new(create_info.model_token)?;

		let mut out_error_msg: *const std::ffi::c_char = std::ptr::null();

		let runtime = unsafe {
			sys::tflite_runtime_create(
				create_info.model_data.as_ptr() as *const i8,
				create_info.model_data.len(),
				delegate_serialization_dir_cstr.as_ptr(),
				model_token_cstr.as_ptr(),
				tflite_warn_log_callback,
				tflite_error_log_callback,
				create_info
					.npu_config
					.as_ref()
					.map(|c| c.config_type)
					.unwrap_or_default()
					.to_u8_constant(),
				create_info.npu_config.is_some(),
				create_info
					.npu_config
					.as_ref()
					.map(|c| c.skel_library_dir.as_ptr())
					.unwrap_or(c"".as_ptr()),
				&mut out_error_msg,
			)
		};

		if runtime.is_null() {
			unsafe {
				Err(handle_error_msg_ptr(out_error_msg)
					.unwrap_or(TfLiteError(
						"no error msg was set, ptr still null".to_string(),
					))
					.into())
			}
		} else {
			Ok(Self {
				runtime,
				model_data: create_info.model_data,
				delegate_serialization_dir: delegate_serialization_dir_cstr,
				model_token: model_token_cstr,
				model_input_format: create_info.model_input_format,
				model_output_format: create_info.model_output_format,
			})
		}
	}

	pub fn run_inference(
		&mut self,
		input: &mut FloatTensorBuffer,
		output: &mut FloatTensorBuffer,
	) -> Result<(), TfLiteError> {
		check_float_tensor_format!(input, self.model_input_format);

		self.run_inference_raw(input.data_mut(), output.data_mut())?;

		output.convert_format(self.model_output_format);

		Ok(())
	}

	fn run_inference_raw(
		&mut self,
		input: &mut [f32],
		output: &mut [f32],
	) -> Result<(), TfLiteError> {
		let mut out_error_msg: *const std::ffi::c_char = std::ptr::null();

		unsafe {
			sys::tflite_runtime_run_inference(
				self.runtime,
				input.as_mut_ptr(),
				input.len(),
				output.as_mut_ptr(),
				output.len(),
				&mut out_error_msg,
			);

			match handle_error_msg_ptr(out_error_msg) {
				Some(e) => Err(e),
				None => Ok(()),
			}
		}
	}

	pub fn get_input_shape(&self) -> &[i32] {
		let mut out_input_shape_ptr = std::ptr::null();
		let mut out_input_shape_len = 0;

		unsafe {
			sys::tflite_runtime_get_input_shape(
				self.runtime,
				&mut out_input_shape_ptr,
				&mut out_input_shape_len,
			);

			std::slice::from_raw_parts(out_input_shape_ptr, out_input_shape_len)
		}
	}

	pub fn get_output_shape(&self) -> &[i32] {
		let mut out_output_shape_ptr = std::ptr::null();
		let mut out_output_shape_len = 0;

		unsafe {
			sys::tflite_runtime_get_output_shape(
				self.runtime,
				&mut out_output_shape_ptr,
				&mut out_output_shape_len,
			);

			std::slice::from_raw_parts(out_output_shape_ptr, out_output_shape_len)
		}
	}

	pub fn allocate_output_tensor(&self) -> FloatTensorBuffer<'static> {
		let output_elements = self.get_output_shape().iter().product::<i32>() as usize;
		let output_container = vec![0.0f32; output_elements];
		FloatTensorBuffer::new(output_container, self.model_output_format)
	}
}
impl Drop for TfLiteRuntime {
	fn drop(&mut self) {
		unsafe {
			sys::tflite_runtime_destroy(self.runtime);
		}
	}
}

/// frees the allocated error msg and returns a rust owned CString
unsafe fn handle_error_msg_ptr(error_msg_ptr: *const std::ffi::c_char) -> Option<TfLiteError> {
	if error_msg_ptr.is_null() {
		None
	} else {
		unsafe {
			let owned = CStr::from_ptr(error_msg_ptr).to_owned();
			sys::tflite_runtime_free_error_msg(error_msg_ptr);
			Some(TfLiteError(
				owned
					.to_str()
					.unwrap_or("invalid cstring tflite error msg")
					.to_string(),
			))
		}
	}
}

extern "C" fn tflite_warn_log_callback(error_msg: *const std::ffi::c_char) {
	let error_msg = unsafe {
		CStr::from_ptr(error_msg)
			.to_str()
			.unwrap_or("invalid cstring tflite warning msg")
	};

	warn!("[TFLITE] {error_msg}");
}
extern "C" fn tflite_error_log_callback(error_msg: *const std::ffi::c_char) {
	let error_msg = unsafe {
		CStr::from_ptr(error_msg)
			.to_str()
			.unwrap_or("invalid cstring tflite error msg")
	};

	error!("[TFLITE] {error_msg}");
}
