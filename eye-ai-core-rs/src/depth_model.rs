use crate::{
	FloatTensorBuffer, FloatTensorFormat, ProfilingFrame, check_float_tensor_format,
	tensor_buffer::WrongFloatTensorFormatError,
	tflite::{
		CreateTfLiteRuntimeInfo, NpuConfig, NpuConfigType, TfLiteRunInferenceError, TfLiteRuntime,
		TfLiteRuntimeCreateError,
	},
};
use eye_ai_core_rs_profiling_attribute::profile_function;
use std::path::PathBuf;
use std::sync::Arc;

pub struct DepthModelNpuConfig {
	pub tflite_qnn_npu_delegate_lib_filepath: PathBuf,
	pub skel_library_dir: std::ffi::CString,
}

pub struct CreateDepthModelInfo {
	pub tflite_lib_filepath: PathBuf,
	/// if None, we try to load gpu delegate api from the tflite_lib_filepath library
	pub tflite_gpu_delegate_lib_filepath: Option<PathBuf>,
	pub model_data: Vec<u8>,
	pub gpu_delegate_serialization_dir: std::ffi::CString,
	pub model_token: std::ffi::CString,
	pub log_warning_callback: Arc<dyn Fn(&str) + Send + Sync>,
	pub log_error_callback: fn(msg: *const std::os::raw::c_char),
	pub npu_config: Option<DepthModelNpuConfig>,
}

#[derive(Debug)]
pub struct DepthModel<'a> {
	runtime: TfLiteRuntime<'a>,
	profiling_frame: &'a ProfilingFrame,
}

impl<'a> DepthModel<'a> {
	pub fn new(
		create_info: CreateDepthModelInfo,
		profiling_frame: &'a ProfilingFrame,
	) -> Result<Self, TfLiteRuntimeCreateError> {
		let runtime_create_info = CreateTfLiteRuntimeInfo {
			tflite_lib_filepath: create_info.tflite_lib_filepath,
			tflite_gpu_delegate_lib_filepath: create_info.tflite_gpu_delegate_lib_filepath,
			model_data: create_info.model_data,
			gpu_delegate_serialization_dir: create_info.gpu_delegate_serialization_dir,
			model_token: create_info.model_token,
			model_input_format: FloatTensorFormat::MiDaSImageRgb,
			model_output_format: FloatTensorFormat::RawRelativeDepth,
			log_warning_callback: create_info.log_warning_callback,
			log_error_callback: create_info.log_error_callback,
			npu_config: create_info.npu_config.map(|depth_npu_config| NpuConfig {
				tflite_qnn_npu_delegate_lib_filepath: depth_npu_config
					.tflite_qnn_npu_delegate_lib_filepath,
				skel_library_dir: depth_npu_config.skel_library_dir,
				config_type: NpuConfigType::MiDaS,
			}),
		};

		let runtime = TfLiteRuntime::new(runtime_create_info, profiling_frame)?;

		Ok(Self {
			runtime,
			profiling_frame,
		})
	}

	/// input format: FloatTensorFormat::MiDaSImageRgb, output format will be: FloatTensorFormat::RawRelativeDepth
	#[profile_function("self.profiling_frame")]
	pub fn run_raw(
		&mut self,
		input_tensor: &FloatTensorBuffer,
		output_tensor: &mut FloatTensorBuffer,
	) -> Result<(), TfLiteRunInferenceError> {
		check_float_tensor_format!(input_tensor, FloatTensorFormat::MiDaSImageRgb);

		self.runtime.run_inference(input_tensor, output_tensor)?;

		check_float_tensor_format!(output_tensor, FloatTensorFormat::RawRelativeDepth);

		Ok(())
	}

	/// input format: FloatTensorFormat::MiDaSImageRgb, output format: FloatTensorFormat::RelativeDepth
	#[profile_function("self.profiling_frame")]
	pub fn run(
		&mut self,
		input_tensor: &mut FloatTensorBuffer,
		output_tensor: &mut FloatTensorBuffer,
	) -> Result<(), TfLiteRunInferenceError> {
		let profiling_frame = self.profiling_frame;

		self.run_raw(input_tensor, output_tensor)?;

		min_max_scaling_operator(output_tensor, profiling_frame)?;

		Ok(())
	}

	#[profile_function("self.profiling_frame")]
	pub fn allocate_output_tensor(
		&self,
	) -> Result<FloatTensorBuffer<'static>, TfLiteRunInferenceError> {
		self.runtime.allocate_output_tensor()
	}

	pub fn get_input_shape(&self) -> Option<Vec<i32>> {
		self.runtime.get_input_shape()
	}

	pub fn get_output_shape(&self) -> Option<Vec<i32>> {
		self.runtime.get_output_shape()
	}
}

/// converts FloatTensorFormat::RawRelativeDepth to FloatTensorFormat::RelativeDepth
#[profile_function("profiling_frame")]
fn min_max_scaling_operator<'a>(
	raw_relative_depth_tensor: &mut FloatTensorBuffer<'a>,
	profiling_frame: &ProfilingFrame,
) -> Result<(), WrongFloatTensorFormatError> {
	check_float_tensor_format!(
		raw_relative_depth_tensor,
		FloatTensorFormat::RawRelativeDepth
	);

	raw_relative_depth_tensor.convert_format(FloatTensorFormat::RelativeDepth);

	if raw_relative_depth_tensor.data().is_empty() {
		return Ok(());
	}

	let min = raw_relative_depth_tensor
		.data()
		.iter()
		.min_by(|a, b| a.total_cmp(b))
		.copied()
		.unwrap();
	let max = raw_relative_depth_tensor
		.data()
		.iter()
		.max_by(|a, b| a.total_cmp(b))
		.copied()
		.unwrap();
	let diff = max - min;
	if diff == 0.0 {
		for value in raw_relative_depth_tensor.iter_mut() {
			*value = 0.5;
		}
	} else {
		for value in raw_relative_depth_tensor.iter_mut() {
			*value = (*value - min) / diff;
		}
	}

	Ok(())
}
