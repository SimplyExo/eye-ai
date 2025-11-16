use crate::{FloatTensorBuffer, ProfilingFrame, TensorFormat, get_tensor_format_name};
use eye_ai_core_rs_profiling_attribute::profile_function;
use semver::VersionReq;
use std::path::PathBuf;
use std::sync::Arc;
use tflite_dyn::{
	Delegate, Interpreter, Model, TfLite,
	sys::{
		GpuDelegateAPI, LoadQnnDelegateLibError, LoadTfLiteGpuDelegateError, QnnDelegateVt,
		TfLiteGpuExperimentalFlags, TfLiteGpuInferenceUsage, TfLiteQnnDelegateBackendType,
		TfLiteQnnDelegateGraphPriority, TfLiteQnnDelegateHtpPerformanceMode,
		TfLiteQnnDelegateHtpPrecision, TfLiteType,
	},
};
use thiserror::Error;
use zerocopy::IntoBytes;

#[derive(Debug, Error)]
pub enum TfLiteRunInferenceError {
	#[error("input tensor not allocated")]
	InputTensorNotAllocated,
	#[error("model input tensor size is {model_expected}, but {provided} was provided")]
	InputTensorSizeMismatch {
		provided: usize,
		model_expected: usize,
	},
	#[error("we only support float32 input tensor's for now")]
	NonFloat32InputTensor,
	#[error("output tensor not allocated")]
	OutputTensorNotAllocated,
	#[error("model output tensor size is {model_expected}, but {provided} was provided")]
	OutputTensorSizeMismatch {
		provided: usize,
		model_expected: usize,
	},
	#[error("we only support float32 output tensor's for now")]
	NonFloat32OutputTensor,
	#[error("failed to invoke inference")]
	Invoke(#[from] tflite_dyn::Error),
	#[error(
		"model expected input of format {}, but {} was provided",
		get_tensor_format_name(*model_expected),
		get_tensor_format_name(*provided)
	)]
	InputFormatMismatch {
		model_expected: TensorFormat,
		provided: TensorFormat,
	},
	#[error(
		"model expected output of format {}, but {} was provided",
		get_tensor_format_name(*model_expected),
		get_tensor_format_name(*provided)
	)]
	OutputFormatMismatch {
		model_expected: TensorFormat,
		provided: TensorFormat,
	},
}

#[derive(Debug, Error)]
pub enum TfLiteRuntimeCreateError {
	#[error("TfLite error: {0}")]
	TfLiteDyn(#[from] tflite_dyn::Error),
	#[error("Failed to load gpu delegate api: {0}")]
	LoadGpuDelegateAPI(#[from] LoadTfLiteGpuDelegateError),
	#[error("Failed to load qnn delegate api: {0}")]
	LoadQnnDelegateAPI(#[from] LoadQnnDelegateLibError),
}

pub enum NpuConfigType {
	MiDaS,
	Yolo,
}

pub struct NpuConfig {
	pub config_type: NpuConfigType,
	pub tflite_qnn_npu_delegate_lib_filepath: PathBuf,
	pub skel_library_dir: std::ffi::CString,
}

pub struct CreateTfLiteRuntimeInfo {
	pub tflite_lib_filepath: PathBuf,
	/// if None, we try to load gpu delegate api from the tflite_lib_filepath library
	pub tflite_gpu_delegate_lib_filepath: Option<PathBuf>,
	pub model_data: Vec<u8>,
	pub gpu_delegate_serialization_dir: std::ffi::CString,
	pub model_token: std::ffi::CString,
	pub model_input_format: TensorFormat,
	pub model_output_format: TensorFormat,
	pub log_warning_callback: Arc<dyn Fn(&str) + Send + Sync>,
	pub log_error_callback: fn(msg: *const std::os::raw::c_char),
	pub npu_config: Option<NpuConfig>,
}

struct CreateTfLiteInterpreterResult {
	interpreter: Interpreter,
	gpu_delegate: Option<Arc<Delegate>>,
	npu_delegate_and_api: Option<(Arc<Delegate>, Arc<QnnDelegateVt>)>,
}

#[allow(unused)]
pub struct TfLiteRuntime<'a> {
	_libabsl_log_internal_nullguard: Result<GlobalLibrary, libloading::Error>,
	tflite_api: Arc<TfLite>,
	tflite_gpu_delegate_api: Arc<GpuDelegateAPI>,
	model_input_format: TensorFormat,
	model_output_format: TensorFormat,
	model: Arc<Model>,
	interpreter: Interpreter,
	gpu_delegate: Option<Arc<Delegate>>,
	npu_delegate_and_api: Option<(Arc<Delegate>, Arc<QnnDelegateVt>)>,
	log_warning_callback: Arc<dyn Fn(&str) + Send + Sync>,
	gpu_delegate_serialization_dir: std::ffi::CString,
	model_token: std::ffi::CString,
	skel_library_dir: Option<std::ffi::CString>,
	profiling_frame: &'a ProfilingFrame,
}
impl<'a> TfLiteRuntime<'a> {
	pub fn new(
		create_info: CreateTfLiteRuntimeInfo,
		profiling_frame: &'a ProfilingFrame,
	) -> Result<Self, TfLiteRuntimeCreateError> {
		let _libabsl_log_internal_nullguard =
			unsafe { load_library_global("libabsl_log_internal_nullguard.so") };

		let tflite_api = Arc::new(
			TfLite::load(
				create_info.tflite_lib_filepath,
				VersionReq::parse(">=2.19.0, <=2.20.0-dev0").unwrap(),
			)
			.map_err(TfLiteRuntimeCreateError::TfLiteDyn)?,
		);
		let tflite_gpu_delegate_api =
			Arc::new(match &create_info.tflite_gpu_delegate_lib_filepath {
				Some(tflite_gpu_delegate_lib_filepath) => {
					GpuDelegateAPI::load_from_separate_library(
						tflite_api.vt.clone(),
						tflite_gpu_delegate_lib_filepath,
					)?
				}
				None => GpuDelegateAPI::load(tflite_api.vt.clone(), tflite_api.vt.library.clone())?,
			});
		let npu_config = match create_info.npu_config {
			Some(npu_config) => {
				let qnn_delegate_api = Arc::new(QnnDelegateVt::load(
					tflite_api.vt.clone(),
					&npu_config.tflite_qnn_npu_delegate_lib_filepath,
				)?);

				Some((npu_config, qnn_delegate_api))
			}
			None => None,
		};

		let model = Arc::new(tflite_api.model_create(create_info.model_data)?);

		let CreateTfLiteInterpreterResult {
			mut interpreter,
			gpu_delegate,
			npu_delegate_and_api,
		} = try_to_create_interpreter(
			&tflite_api,
			&tflite_gpu_delegate_api,
			model.clone(),
			&create_info.gpu_delegate_serialization_dir,
			&create_info.model_token,
			npu_config
				.as_ref()
				.map(|(npu_config, qnn_delegate)| (npu_config, qnn_delegate.clone())),
			&*create_info.log_warning_callback,
			create_info.log_error_callback as *mut std::os::raw::c_void,
			profiling_frame,
		)?;

		interpreter.allocate_tensors()?;

		Ok(Self {
			_libabsl_log_internal_nullguard,
			tflite_api,
			tflite_gpu_delegate_api,
			interpreter,
			model,
			model_input_format: create_info.model_input_format,
			model_output_format: create_info.model_output_format,
			gpu_delegate,
			npu_delegate_and_api,
			log_warning_callback: create_info.log_warning_callback,
			gpu_delegate_serialization_dir: create_info.gpu_delegate_serialization_dir,
			model_token: create_info.model_token,
			skel_library_dir: npu_config.map(|(npu_config, _)| npu_config.skel_library_dir),
			profiling_frame,
		})
	}

	pub fn get_input_shape(&self) -> Option<Vec<i32>> {
		let input_tensor = self.interpreter.input_tensor(0)?;
		let mut input_shape = Vec::with_capacity(input_tensor.num_dims() as usize);

		for i in 0..input_shape.capacity() {
			input_shape.push(input_tensor.dim(i as i32));
		}

		Some(input_shape)
	}

	pub fn get_output_shape(&self) -> Option<Vec<i32>> {
		let output_tensor = self.interpreter.output_tensor(0)?;
		let mut output_shape = Vec::with_capacity(output_tensor.num_dims() as usize);

		for i in 0..output_shape.capacity() {
			output_shape.push(output_tensor.dim(i as i32));
		}

		Some(output_shape)
	}

	#[profile_function("self.profiling_frame")]
	pub fn run_inference(
		&mut self,
		input: &[f32],
		output: &mut [f32],
	) -> Result<(), TfLiteRunInferenceError> {
		let mut input_tensor = self
			.interpreter
			.input_tensor(0)
			.ok_or(TfLiteRunInferenceError::InputTensorNotAllocated)?;
		if input_tensor.type_() != TfLiteType::Float32 {
			return Err(TfLiteRunInferenceError::NonFloat32InputTensor);
		}
		let input_tensor_data = input_tensor
			.data_mut()
			.ok_or(TfLiteRunInferenceError::InputTensorNotAllocated)?;
		if input_tensor_data.len() != std::mem::size_of_val(input) {
			return Err(TfLiteRunInferenceError::InputTensorSizeMismatch {
				provided: std::mem::size_of_val(input),
				model_expected: input_tensor_data.len(),
			});
		}
		copy_f32_to_tensor(input, input_tensor_data, self.profiling_frame);

		self.interpreter.invoke()?;

		let mut output_tensor = self
			.interpreter
			.output_tensor(0)
			.ok_or(TfLiteRunInferenceError::OutputTensorNotAllocated)?;
		let output_tensor_data = output_tensor
			.data_mut()
			.ok_or(TfLiteRunInferenceError::OutputTensorNotAllocated)?;
		if output_tensor_data.len() != std::mem::size_of_val(output) {
			return Err(TfLiteRunInferenceError::OutputTensorSizeMismatch {
				provided: std::mem::size_of_val(output),
				model_expected: output_tensor_data.len(),
			});
		}
		if output_tensor.type_() != TfLiteType::Float32 {
			return Err(TfLiteRunInferenceError::NonFloat32OutputTensor);
		}

		copy_f32_from_tensor(output_tensor_data, output, self.profiling_frame);

		Ok(())
	}

	#[profile_function("self.profiling_frame")]
	pub fn run_inference_with_tensors<
		const INPUT_FORMAT: TensorFormat,
		const OUTPUT_FORMAT: TensorFormat,
	>(
		&mut self,
		input_tensor: FloatTensorBuffer<INPUT_FORMAT>,
	) -> Result<FloatTensorBuffer<'_, OUTPUT_FORMAT>, TfLiteRunInferenceError> {
		if self.model_input_format != INPUT_FORMAT {
			return Err(TfLiteRunInferenceError::InputFormatMismatch {
				model_expected: self.model_input_format,
				provided: INPUT_FORMAT,
			});
		}
		if self.model_output_format != OUTPUT_FORMAT {
			return Err(TfLiteRunInferenceError::OutputFormatMismatch {
				model_expected: self.model_output_format,
				provided: OUTPUT_FORMAT,
			});
		}

		let mut output_container = Vec::<f32>::new();
		let output_tensor = self
			.interpreter
			.output_tensor(0)
			.ok_or(TfLiteRunInferenceError::OutputTensorNotAllocated)?;
		let output_tensor_data = output_tensor
			.data()
			.ok_or(TfLiteRunInferenceError::OutputTensorNotAllocated)?;
		output_container.resize(
			output_tensor_data.len() / std::mem::size_of::<f32>(),
			0.0f32,
		);
		self.run_inference(input_tensor.data(), &mut output_container)?;
		Ok(FloatTensorBuffer::<'_, OUTPUT_FORMAT>::new(
			output_container,
		))
	}
}

unsafe extern "C" {
	unsafe fn tflite_error_callback(
		user_data_ptr: *mut std::os::raw::c_void,
		format: *const std::os::raw::c_char,
		args: va_list::VaList,
	);
}

#[allow(clippy::too_many_arguments)]
#[profile_function("profiling_frame")]
fn try_to_create_interpreter<'a>(
	tflite_api: &'a TfLite,
	tflite_gpu_delegate_api: &'a GpuDelegateAPI,
	model: Arc<Model>,
	delegate_serialization_dir: &'a std::ffi::CStr,
	model_token: &'a std::ffi::CStr,
	npu_config: Option<(&'a NpuConfig, Arc<QnnDelegateVt>)>,
	log_warning_callback: impl Fn(&str),
	error_reporter_user_data_ptr: *mut std::os::raw::c_void,
	profiling_frame: &ProfilingFrame,
) -> Result<CreateTfLiteInterpreterResult, tflite_dyn::Error> {
	let mut interpreter_options = tflite_api.interpreter_options_create();
	interpreter_options.set_num_threads(4);
	unsafe {
		interpreter_options.set_error_reporter(tflite_error_callback, error_reporter_user_data_ptr);
	}

	if let Some((npu_config, qnn_delegate_api)) = npu_config {
		let npu_delegate = create_npu_delegate(
			&qnn_delegate_api,
			delegate_serialization_dir,
			model_token,
			npu_config,
			profiling_frame,
		);

		log_warning_callback("QNN NPU delegate was created!");

		let npu_delegate = Arc::new(npu_delegate);
		let mut interpreter_options_with_npu_delegate = interpreter_options.clone();
		interpreter_options_with_npu_delegate.add_delegate(npu_delegate.clone());

		let interpreter_result =
			tflite_api.interpreter_create(model.clone(), interpreter_options_with_npu_delegate);

		match interpreter_result {
			Some(interpreter) => {
				return Ok(CreateTfLiteInterpreterResult {
					interpreter,
					gpu_delegate: None,
					npu_delegate_and_api: Some((npu_delegate, qnn_delegate_api)),
				});
			}
			None => {
				log_warning_callback(
					"NPU Delegate is not supported, trying GPU Delegate support next!",
				);
			}
		}
	} else {
		log_warning_callback("No QNN NPU delegate was created!");
	}

	let gpu_delegate = create_gpu_delegate(
		tflite_gpu_delegate_api,
		delegate_serialization_dir,
		model_token,
		profiling_frame,
	);

	let gpu_delegate = Arc::new(gpu_delegate);
	let mut interpreter_options_with_gpu_delegate = interpreter_options.clone();
	interpreter_options_with_gpu_delegate.add_delegate(gpu_delegate.clone());
	if let Some(interpreter) =
		tflite_api.interpreter_create(model.clone(), interpreter_options_with_gpu_delegate)
	{
		return Ok(CreateTfLiteInterpreterResult {
			interpreter,
			gpu_delegate: Some(gpu_delegate),
			npu_delegate_and_api: None,
		});
	}

	match tflite_api.interpreter_create(model, interpreter_options) {
		Some(interpreter) => Ok(CreateTfLiteInterpreterResult {
			interpreter,
			gpu_delegate: None,
			npu_delegate_and_api: None,
		}),
		None => Err(tflite_dyn::Error::Generic),
	}
}

#[profile_function("profiling_frame")]
fn create_gpu_delegate<'a>(
	tflite_gpu_delegate_api: &'a GpuDelegateAPI,
	gpu_delegate_serialization_dir: &'a std::ffi::CStr,
	model_token: &'a std::ffi::CStr,
	profiling_frame: &ProfilingFrame,
) -> Delegate {
	let mut gpu_options_v2 = tflite_gpu_delegate_api.default_options();
	gpu_options_v2.is_precision_loss_allowed = true as i32;
	gpu_options_v2.inference_preference = TfLiteGpuInferenceUsage::FastSingleAnswer;
	gpu_options_v2.experimental_flags |= TfLiteGpuExperimentalFlags::EnableSerialization as i64;
	gpu_options_v2.serialization_dir = gpu_delegate_serialization_dir.as_ptr();
	gpu_options_v2.model_token = model_token.as_ptr();
	tflite_gpu_delegate_api.create_delegate(&gpu_options_v2)
}

#[profile_function("profiling_frame")]
fn create_npu_delegate<'a>(
	tflite_npu_delegate_api: &'a QnnDelegateVt,
	npu_delegate_serialization_dir: &'a std::ffi::CStr,
	model_token: &'a std::ffi::CStr,
	config: &'a NpuConfig,
	profiling_frame: &ProfilingFrame,
) -> Delegate {
	let mut options = tflite_npu_delegate_api.default_options();
	options.cache_dir = npu_delegate_serialization_dir.as_ptr();
	options.model_token = model_token.as_ptr();
	options.graph_priority = TfLiteQnnDelegateGraphPriority::High;
	options.backend_type = TfLiteQnnDelegateBackendType::Htp;
	options.skel_library_path = config.skel_library_dir.as_ptr();
	options.htp_options.use_conv_hmx = false;

	match config.config_type {
		NpuConfigType::MiDaS => {
			options.htp_options.precision = TfLiteQnnDelegateHtpPrecision::Fp16;
			options.htp_options.performance_mode = TfLiteQnnDelegateHtpPerformanceMode::Burst;
		}
		NpuConfigType::Yolo => {
			options.htp_options.precision = TfLiteQnnDelegateHtpPrecision::Quantized;
			options.htp_options.performance_mode =
				TfLiteQnnDelegateHtpPerformanceMode::SustainedHighPerformance;
		}
	}

	tflite_npu_delegate_api.create_delegate(&options)
}

#[profile_function("profiling_frame")]
fn copy_f32_to_tensor(
	src_f32: &[f32],
	dst_tensor_bytes: &mut [u8],
	profiling_frame: &ProfilingFrame,
) {
	let src_bytes = src_f32.as_bytes();
	let copy_bytes_len = src_bytes.len();
	assert_eq!(copy_bytes_len, dst_tensor_bytes.len());
	dst_tensor_bytes.copy_from_slice(src_bytes);
}

#[profile_function("profiling_frame")]
fn copy_f32_from_tensor(
	src_tensor_bytes: &[u8],
	dst_f32: &mut [f32],
	profiling_frame: &ProfilingFrame,
) {
	let dst_bytes = dst_f32.as_mut_bytes();
	let copy_bytes_len = src_tensor_bytes.len();
	assert_eq!(copy_bytes_len, dst_bytes.len());
	dst_bytes.copy_from_slice(src_tensor_bytes);
}

#[cfg(target_family = "unix")]
type GlobalLibrary = libloading::os::unix::Library;

#[cfg(not(target_family = "unix"))]
type GlobalLibrary = libloading::Library;

#[cfg(target_family = "unix")]
unsafe fn load_library_global<P: AsRef<std::ffi::OsStr>>(
	path: P,
) -> Result<GlobalLibrary, libloading::Error> {
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
) -> Result<GlobalLibrary, libloading::Error> {
	unsafe { Library::new(path) }
}
