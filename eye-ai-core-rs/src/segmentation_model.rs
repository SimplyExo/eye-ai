use eye_ai_core_rs_profiling_attribute::profile_function;
use rayon::prelude::*;
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

	let input_pixels = input.data().as_chunks::<3>().0;

	let mut output = FloatTensorBuffer::new(
		vec![0.0; input.data().len()],
		FloatTensorFormat::YoloImageRgb,
	);

	let output_data = output.data_mut();

	let plane = width * height;

	// 0.0..255.0 -> 0.0..1.0 + HWC -> CHW
	let (r_plane, rest) = output_data.split_at_mut(plane);
	let (g_plane, b_plane) = rest.split_at_mut(plane);

	let inv_255 = 1.0 / 255.0;

	// split all three output planes into the same per-worker ranges, then
	// let each worker write a disjoint set of output pixels in parallel
	let num_workers = rayon::current_num_threads();
	let chunk_size = plane.div_ceil(num_workers);

	let plane_chunks: Vec<(&mut [f32], &mut [f32], &mut [f32])> = r_plane
		.chunks_mut(chunk_size)
		.zip(g_plane.chunks_mut(chunk_size))
		.zip(b_plane.chunks_mut(chunk_size))
		.map(|((r, g), b)| (r, g, b))
		.collect();

	plane_chunks
		.into_par_iter()
		.enumerate()
		.for_each(|(chunk_index, (r, g, b))| {
			let start = chunk_index * chunk_size;
			for (j, ((r_out, g_out), b_out)) in
				r.iter_mut().zip(g.iter_mut()).zip(b.iter_mut()).enumerate()
			{
				let pixel = &input_pixels[start + j];
				*r_out = pixel[0] * inv_255;
				*g_out = pixel[1] * inv_255;
				*b_out = pixel[2] * inv_255;
			}
		});

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
	let plane = width * height;

	if plane == 0 {
		return;
	}

	// iterate class plane by class plane (contiguous reads) and keep a running
	// max + argmax per pixel, instead of striding over classes per pixel
	let (_, max_class_indices) =
		crate::argmax::argmax_over_planes(tensor_data, plane, num_classes, f32::MIN);

	assert!(plane <= postprocessed.len());
	postprocessed[..plane].copy_from_slice(&max_class_indices);
}
