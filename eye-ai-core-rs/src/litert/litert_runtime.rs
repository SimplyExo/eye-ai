use crate::{
	FloatTensorBuffer, FloatTensorFormat, ProfilingFrame,
	tensor_buffer::WrongFloatTensorFormatError,
};
use eye_ai_core_rs_profiling_attribute::profile_function;
use litert::{
	Accelerators, CompilationOptions, CompiledModel, ElementType, Environment, Model, TensorBuffer,
	TensorShape,
};
use thiserror::Error;
use tracing::debug;

#[derive(Debug, Error)]
pub enum LiteRtRunInferenceError {
	#[error("model input tensor size is {model_expected}, but {provided} was provided")]
	InputTensorSizeMismatch {
		provided: usize,
		model_expected: usize,
	},
	#[error("model output tensor size is {model_expected}, but {provided} was provided")]
	OutputTensorSizeMismatch {
		provided: usize,
		model_expected: usize,
	},
	#[error("failed to invoke inference")]
	Invoke(#[from] litert::Error),
	#[error(
		"model expected input of format {}, but {} was provided",
		model_expected,
		provided
	)]
	InputFormatMismatch {
		model_expected: FloatTensorFormat,
		provided: FloatTensorFormat,
	},
	#[error("wrong format while converting: {0}")]
	WrongFormatForConvertion(#[from] WrongFloatTensorFormatError),
	#[error("failed to query output shape from compiled model")]
	OutputShapeQuery,
	#[error("failed to query input shape from compiled model")]
	InputShapeQuery,
}

#[derive(Debug, Error)]
pub enum LiteRtRuntimeCreateError {
	#[error("LiteRT error: {0}")]
	LiteRt(#[from] litert::Error),
	#[error("failed to query model signature")]
	NoSignature,
}

#[derive(Debug)]
pub enum NpuConfigType {
	MiDaS,
	Yolo,
}

#[derive(Debug)]
pub struct NpuConfig {
	pub config_type: NpuConfigType,
	pub skel_library_dir: std::ffi::CString,
}

pub struct CreateLiteRtRuntimeInfo {
	pub model_name: String,
	pub model_data: Vec<u8>,
	pub model_input_format: FloatTensorFormat,
	pub model_output_format: FloatTensorFormat,
	pub npu_config: Option<NpuConfig>,
}

pub struct LiteRtRuntime<'a> {
	_env: Environment,
	_model: Model,
	model_name: String,
	compiled: CompiledModel,
	model_input_format: FloatTensorFormat,
	model_output_format: FloatTensorFormat,
	_profiling_frame: &'a ProfilingFrame,
	input_shape: Vec<i32>,
	output_shape: Vec<i32>,
	input_num_elements: usize,
	output_num_elements: usize,
}
impl<'a> std::fmt::Debug for LiteRtRuntime<'a> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("LiteRtRuntime")
			.field("model_name", &self.model_name)
			.field("model_input_format", &self.model_input_format)
			.field("model_output_format", &self.model_output_format)
			.field("input_shape", &self.input_shape)
			.field("output_shape", &self.output_shape)
			.field("input_num_elements", &self.input_num_elements)
			.field("output_num_elements", &self.output_num_elements)
			.field("profiling_frame", &self._profiling_frame)
			.finish_non_exhaustive()
	}
}
// Safety: CompiledModel::run() takes &self and the LiteRT C runtime handles
// internal serialisation. Our LiteRtRuntime is always behind a RwLock.
unsafe impl<'a> Sync for LiteRtRuntime<'a> {}

impl<'a> LiteRtRuntime<'a> {
	#[profile_function("profiling_frame")]
	pub fn new(
		create_info: CreateLiteRtRuntimeInfo,
		profiling_frame: &'a ProfilingFrame,
	) -> Result<Self, LiteRtRuntimeCreateError> {
		let env = Environment::new()?;
		let model = Model::from_bytes(create_info.model_data)?;

		let compiled = compile_model(
			&create_info.model_name,
			&model,
			create_info.npu_config.is_some(),
			profiling_frame,
		)?;

		let sig = model
			.signature(0)
			.map_err(|_| LiteRtRuntimeCreateError::NoSignature)?;
		let in_shape = sig.input_shape(0)?;
		let out_shape = sig.output_shape(0)?;

		let input_num_elements = in_shape.num_elements();
		let output_num_elements = out_shape.num_elements();

		Ok(Self {
			_env: env,
			_model: model,
			model_name: create_info.model_name,
			compiled,
			model_input_format: create_info.model_input_format,
			model_output_format: create_info.model_output_format,
			_profiling_frame: profiling_frame,
			input_shape: in_shape.dims,
			output_shape: out_shape.dims,
			input_num_elements,
			output_num_elements,
		})
	}

	pub fn get_input_shape(&self) -> Option<Vec<i32>> {
		Some(self.input_shape.clone())
	}

	pub fn get_output_shape(&self) -> Option<Vec<i32>> {
		Some(self.output_shape.clone())
	}

	/// output will be formatted with self.model_output_format
	#[profile_function("self._profiling_frame")]
	pub fn run_inference(
		&self,
		input: &FloatTensorBuffer,
		output: &mut FloatTensorBuffer,
	) -> Result<(), LiteRtRunInferenceError> {
		if self.model_input_format != input.format() {
			return Err(LiteRtRunInferenceError::InputFormatMismatch {
				model_expected: self.model_input_format,
				provided: input.format(),
			});
		}

		let provided_input_len = input.data().len();
		if provided_input_len != self.input_num_elements {
			return Err(LiteRtRunInferenceError::InputTensorSizeMismatch {
				provided: provided_input_len,
				model_expected: self.input_num_elements,
			});
		}

		let provided_output_len = output.data().len();
		if provided_output_len != self.output_num_elements {
			return Err(LiteRtRunInferenceError::OutputTensorSizeMismatch {
				provided: provided_output_len,
				model_expected: self.output_num_elements,
			});
		}

		let input_shape = TensorShape {
			element_type: ElementType::Float32,
			dims: self.input_shape.clone(),
		};
		let mut input_buffer = TensorBuffer::managed_host(&self._env, &input_shape)?;
		{
			let mut guard = input_buffer.lock_for_write::<f32>()?;
			guard.copy_from_slice(input.data());
		}

		let output_shape = TensorShape {
			element_type: ElementType::Float32,
			dims: self.output_shape.clone(),
		};
		let output_buffer = TensorBuffer::managed_host(&self._env, &output_shape)?;

		let mut inputs = [input_buffer];
		let mut outputs = [output_buffer];

		self.compiled.run(&mut inputs, &mut outputs)?;

		{
			let guard = outputs[0].lock_for_read::<f32>()?;
			output.data_mut().copy_from_slice(&guard);
		}

		output.convert_format(self.model_output_format);

		Ok(())
	}

	/// allocates the output FloatTensorBuffer to automatically fit the model output
	#[profile_function("self._profiling_frame")]
	pub fn allocate_output_tensor(
		&self,
	) -> Result<FloatTensorBuffer<'static>, LiteRtRunInferenceError> {
		let output_container = vec![0.0f32; self.output_num_elements];

		Ok(FloatTensorBuffer::new(
			output_container,
			self.model_output_format,
		))
	}
}

#[profile_function("profiling_frame")]
fn compile_model(
	model_name: &str,
	model: &Model,
	npu_requested: bool,
	profiling_frame: &ProfilingFrame,
) -> Result<CompiledModel, LiteRtRuntimeCreateError> {
	let mut accelerators = Accelerators::CPU | Accelerators::GPU;
	if npu_requested {
		accelerators = accelerators | Accelerators::NPU;
	}

	debug!(
		model_name = ?model_name,
		accelerators = ?format_accelerators(accelerators),
		"compile_model()",
	);

	let env = Environment::new()?;

	// FIX: add needed options when npu is enabled!
	let options = CompilationOptions::new()?.with_accelerators(accelerators)?;

	Ok(CompiledModel::new(env, model.clone(), &options)?)
}

fn format_accelerators(accelerators: Accelerators) -> String {
	let mut accelerators_str = Vec::new();
	if accelerators.contains(Accelerators::CPU) {
		accelerators_str.push("CPU");
	}
	if accelerators.contains(Accelerators::GPU) {
		accelerators_str.push("GPU");
	}
	if accelerators.contains(Accelerators::NPU) {
		accelerators_str.push("NPU");
	}
	accelerators_str.join(", ")
}
