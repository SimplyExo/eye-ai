use eye_ai_core_rs_profiling_attribute::profile_function;

use crate::{
	FLOAT_TENSOR_BUFFER_MIDAS_IMAGE_RGB_FORMAT, FLOAT_TENSOR_BUFFER_RAW_RELATIVE_DEPTH_FORMAT,
	FLOAT_TENSOR_BUFFER_RELATIVE_DEPTH_FORMAT, FloatTensorBuffer,
	tflite::{
		CreateTfLiteRuntimeInfo, NpuConfig, NpuConfigType, TfLiteRunInferenceError, TfLiteRuntime,
		TfLiteRuntimeCreateError,
	},
};
use std::path::PathBuf;

pub struct DepthModelNpuConfig<'a> {
	pub tflite_qnn_npu_delegate_lib_filepath: PathBuf,
	pub skel_library_dir: &'a std::ffi::CStr,
}

pub struct CreateDepthModelInfo<'a> {
	pub tflite_lib_filepath: PathBuf,
	/// if None, we try to load gpu delegate api from the tflite_lib_filepath library
	pub tflite_gpu_delegate_lib_filepath: Option<PathBuf>,
	pub model_data: Vec<u8>,
	pub gpu_delegate_serialization_dir: &'a std::ffi::CStr,
	pub model_token: &'a std::ffi::CStr,
	pub log_warning_callback: fn(msg: &str),
	pub log_error_callback: fn(msg: *const std::os::raw::c_char),
	pub npu_config: Option<DepthModelNpuConfig<'a>>,
}

pub struct DepthModel {
	runtime: TfLiteRuntime,
}

impl DepthModel {
	pub fn new(create_info: CreateDepthModelInfo) -> Result<Self, TfLiteRuntimeCreateError> {
		let runtime_create_info = CreateTfLiteRuntimeInfo {
			tflite_lib_filepath: create_info.tflite_lib_filepath,
			tflite_gpu_delegate_lib_filepath: create_info.tflite_gpu_delegate_lib_filepath,
			model_data: create_info.model_data,
			gpu_delegate_serialization_dir: create_info.gpu_delegate_serialization_dir,
			model_token: create_info.model_token,
			model_input_format: FLOAT_TENSOR_BUFFER_MIDAS_IMAGE_RGB_FORMAT,
			model_output_format: FLOAT_TENSOR_BUFFER_RAW_RELATIVE_DEPTH_FORMAT,
			log_warning_callback: create_info.log_warning_callback,
			log_error_callback: create_info.log_error_callback,
			npu_config: create_info.npu_config.map(|depth_npu_config| NpuConfig {
				tflite_qnn_npu_delegate_lib_filepath: depth_npu_config
					.tflite_qnn_npu_delegate_lib_filepath,
				skel_library_dir: depth_npu_config.skel_library_dir,
				config_type: NpuConfigType::MiDaS,
			}),
		};

		let runtime = TfLiteRuntime::new(runtime_create_info)?;

		Ok(Self { runtime })
	}

	#[profile_function]
	pub fn run_raw(
		&mut self,
		input_tensor: FloatTensorBuffer<FLOAT_TENSOR_BUFFER_MIDAS_IMAGE_RGB_FORMAT>,
	) -> Result<
		FloatTensorBuffer<'_, FLOAT_TENSOR_BUFFER_RAW_RELATIVE_DEPTH_FORMAT>,
		TfLiteRunInferenceError,
	> {
		self.runtime.run_inference_with_tensors(input_tensor)
	}

	#[profile_function]
	pub fn run(
		&mut self,
		input_tensor: FloatTensorBuffer<FLOAT_TENSOR_BUFFER_MIDAS_IMAGE_RGB_FORMAT>,
	) -> Result<
		FloatTensorBuffer<'_, FLOAT_TENSOR_BUFFER_RELATIVE_DEPTH_FORMAT>,
		TfLiteRunInferenceError,
	> {
		let raw_relative_depth_tensor = self.run_raw(input_tensor)?;

		Ok(min_max_scaling_operator(raw_relative_depth_tensor))
	}
}

#[profile_function]
fn min_max_scaling_operator(
	raw_relative_depth_tensor: FloatTensorBuffer<FLOAT_TENSOR_BUFFER_RAW_RELATIVE_DEPTH_FORMAT>,
) -> FloatTensorBuffer<FLOAT_TENSOR_BUFFER_RELATIVE_DEPTH_FORMAT> {
	let mut relative_depth_tensor =
		raw_relative_depth_tensor.convert_format::<FLOAT_TENSOR_BUFFER_RELATIVE_DEPTH_FORMAT>();

	if relative_depth_tensor.data().is_empty() {
		return relative_depth_tensor;
	}

	let min = relative_depth_tensor
		.data()
		.iter()
		.min_by(|a, b| a.total_cmp(b))
		.copied()
		.unwrap();
	let max = relative_depth_tensor
		.data()
		.iter()
		.max_by(|a, b| a.total_cmp(b))
		.copied()
		.unwrap();
	let diff = max - min;
	if diff == 0.0 {
		for value in relative_depth_tensor.iter_mut() {
			*value = 0.5;
		}
	} else {
		for value in relative_depth_tensor.iter_mut() {
			*value = (*value - min) / diff;
		}
	}

	relative_depth_tensor
}
