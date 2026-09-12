use eye_ai_core_rs_profiling_attribute::profile_function;
use tracing::debug;

use crate::{
	FloatTensorBuffer, FloatTensorFormat, ProfilingFrame, check_float_tensor_format,
	tflite_runtime::{
		CreateTfLiteRuntimeError, CreateTfLiteRuntimeInfo, NpuConfig, NpuConfigType, TfLiteError,
		TfLiteRuntime,
	},
};

#[derive(Debug)]
pub struct SegmentationModelNpuConfig {
	pub skel_library_dir: std::ffi::CString,
}

#[derive(Debug)]
pub struct CreateSegmentationModelInfo {
	pub model_name: String,
	pub model_data: Vec<u8>,
	pub classes: Vec<String>,
	pub delegate_serialization_dir: String,
	pub model_token: String,
	pub npu_config: Option<SegmentationModelNpuConfig>,
}

#[derive(Debug)]
pub struct SegmentationModel<'a> {
	runtime: TfLiteRuntime,
	classes: Vec<String>,
	width: usize,
	height: usize,
	profiling_frame: &'a ProfilingFrame,
}
impl<'a> SegmentationModel<'a> {
	#[profile_function("profiling_frame")]
	pub fn new(
		create_info: CreateSegmentationModelInfo,
		profiling_frame: &'a ProfilingFrame,
	) -> Result<Self, CreateTfLiteRuntimeError> {
		debug!(
			model_name = ?create_info.model_name,
			npu_config = ?create_info.npu_config,
			"new()"
		);

		let npu_config = create_info
			.npu_config
			.map(|segmentation_npu_config| NpuConfig {
				skel_library_dir: segmentation_npu_config.skel_library_dir,
				config_type: NpuConfigType::Yolo,
			});

		let runtime = TfLiteRuntime::new(CreateTfLiteRuntimeInfo {
			model_data: create_info.model_data,
			model_input_format: FloatTensorFormat::YoloImageRgb,
			model_output_format: FloatTensorFormat::YoloSegmentationOutput,
			delegate_serialization_dir: create_info.delegate_serialization_dir,
			model_token: create_info.model_token,
			npu_config,
		})?;

		let output_shape = runtime.get_output_shape();
		let width = *output_shape.get(2).unwrap() as usize;
		let height = *output_shape.get(3).unwrap() as usize;

		Ok(Self {
			runtime,
			classes: create_info.classes,
			width,
			height,
			profiling_frame,
		})
	}

	/// input format: FloatTensorFormat::ImageRgb255, output format will be: per pixel class index
	#[profile_function("self.profiling_frame")]
	pub fn run(
		&mut self,
		input_tensor: &mut FloatTensorBuffer,
		output_tensor: &mut [i32],
	) -> Result<(), TfLiteError> {
		check_float_tensor_format!(input_tensor, FloatTensorFormat::ImageRgb255);

		let mut preprocessed =
			preprocess(input_tensor, self.width, self.height, self.profiling_frame);

		let mut raw_output_tensor = self.runtime.allocate_output_tensor();

		self.runtime
			.run_inference(&mut preprocessed, &mut raw_output_tensor)?;

		check_float_tensor_format!(
			&raw_output_tensor,
			FloatTensorFormat::YoloSegmentationOutput
		);

		postprocess(
			&raw_output_tensor,
			output_tensor,
			self.width,
			self.height,
			self.classes.len(),
			self.profiling_frame,
		);

		Ok(())
	}

	pub fn get_input_shape(&self) -> &[i32] {
		self.runtime.get_input_shape()
	}

	pub fn get_output_shape(&self) -> &[i32] {
		self.runtime.get_output_shape()
	}

	pub fn get_classes(&self) -> &[String] {
		&self.classes
	}
}

/// converts a FloatTensorFormat::ImageRgb255 image to FloatTensorFormat::YoloImageRgb
#[profile_function("profiling_frame")]
fn preprocess<'a>(
	input: &'a FloatTensorBuffer<'a>,
	width: usize,
	height: usize,
	profiling_frame: &ProfilingFrame,
) -> FloatTensorBuffer<'static> {
	check_float_tensor_format!(input, FloatTensorFormat::ImageRgb255);

	assert_eq!(input.data().len(), 3 * width * height);

	let mut output = FloatTensorBuffer::new(
		vec![0.0; input.data().len()],
		FloatTensorFormat::YoloImageRgb,
	);

	let output_data = output.data_mut();

	let plane = width * height;

	// 0.0..255.0 -> 0.0..1.0 + HWC -> CHW
	for (i, pixel) in input.data().as_chunks::<3>().0.iter().enumerate() {
		output_data[i] = pixel[0] / 255.0;
		output_data[plane + i] = pixel[1] / 255.0;
		output_data[2 * plane + i] = pixel[2] / 255.0;
	}

	output
}

/// takes a `tensor_data` of format `FloatTensorFormat::YoloSegmentationOutput` (classes, width,
/// height) and returns a (width, height) of class indices
#[profile_function("profiling_frame")]
fn postprocess(
	tensor: &FloatTensorBuffer,
	postprocessed: &mut [i32],
	width: usize,
	height: usize,
	num_classes: usize,
	profiling_frame: &ProfilingFrame,
) {
	check_float_tensor_format!(tensor, FloatTensorFormat::YoloSegmentationOutput);
	assert_eq!(tensor.data().len(), num_classes * width * height);

	let tensor_data = tensor.data();

	for y in 0..height {
		for x in 0..width {
			let mut max = f32::MIN;
			let mut max_class_index: i32 = 0;
			for c in 0..num_classes {
				let cnf = tensor_data[c * width * height + y * width + x];
				if cnf > max {
					max = cnf;
					max_class_index = c as i32;
				}
			}
			postprocessed[y * width + x] = max_class_index;
		}
	}
}
