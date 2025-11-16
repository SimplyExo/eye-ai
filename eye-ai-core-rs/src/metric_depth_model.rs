use eye_ai_core_rs_profiling_attribute::profile_function;

use crate::{
	CreateDepthModelInfo, DepthModel, FLOAT_TENSOR_BUFFER_METRIC_DEPTH_FORMAT,
	FLOAT_TENSOR_BUFFER_MIDAS_IMAGE_RGB_FORMAT, FLOAT_TENSOR_BUFFER_RAW_RELATIVE_DEPTH_FORMAT,
	FloatTensorBuffer,
	profiling::ProfilingFrame,
	tflite::{TfLiteRunInferenceError, TfLiteRuntimeCreateError},
};

pub struct MetricDepthModel<'a> {
	relative_depth_model: DepthModel<'a>,
	profiling_frame: &'a ProfilingFrame,
}
impl<'a> MetricDepthModel<'a> {
	const REL2ABS_COEFFS: [f32; 5] = [4.30595, -6.5995E-03, 5.25059E-6, -2.7962E-9, 9.28594E-13];

	pub fn new(
		relative_depth_model_create_info: CreateDepthModelInfo,
		profiling_frame: &'a ProfilingFrame,
	) -> Result<Self, TfLiteRuntimeCreateError> {
		let relative_depth_model =
			DepthModel::new(relative_depth_model_create_info, profiling_frame)?;

		Ok(Self {
			relative_depth_model,
			profiling_frame,
		})
	}

	#[profile_function("self.profiling_frame")]
	pub fn run(
		&mut self,
		input_tensor: FloatTensorBuffer<FLOAT_TENSOR_BUFFER_MIDAS_IMAGE_RGB_FORMAT>,
	) -> Result<
		FloatTensorBuffer<'_, FLOAT_TENSOR_BUFFER_METRIC_DEPTH_FORMAT>,
		TfLiteRunInferenceError,
	> {
		let relative_depth_tensor = self.relative_depth_model.run_raw(input_tensor)?;

		Ok(rel2abs_operator(
			relative_depth_tensor,
			&Self::REL2ABS_COEFFS,
			self.profiling_frame,
		))
	}

	pub fn get_input_shape(&self) -> Option<Vec<i32>> {
		self.relative_depth_model.get_input_shape()
	}

	pub fn get_output_shape(&self) -> Option<Vec<i32>> {
		self.relative_depth_model.get_output_shape()
	}
}

#[profile_function("profiling_frame")]
fn rel2abs_operator<'a>(
	relative_depth_tensor: FloatTensorBuffer<'a, FLOAT_TENSOR_BUFFER_RAW_RELATIVE_DEPTH_FORMAT>,
	rel2abs_coeffs: &'a [f32; 5],
	profiling_frame: &ProfilingFrame,
) -> FloatTensorBuffer<'a, FLOAT_TENSOR_BUFFER_METRIC_DEPTH_FORMAT> {
	let mut metric_depth_tensor =
		relative_depth_tensor.convert_format::<FLOAT_TENSOR_BUFFER_METRIC_DEPTH_FORMAT>();
	for value in metric_depth_tensor.iter_mut() {
		*value = polynomial_n4(*value, rel2abs_coeffs);
	}
	metric_depth_tensor
}

/// Polynomial function of degree 4 using Horner's method.
///
/// coeffs: {a0, a1, a2, a3, a4}
///
/// polynomial_4(x) = a0 + a1 * x + a2 * x² + a3 * x³ + a4 * x⁴
fn polynomial_n4(x: f32, coeffs: &[f32; 5]) -> f32 {
	let mut y = coeffs[4];
	y = y * x + coeffs[3];
	y = y * x + coeffs[2];
	y = y * x + coeffs[1];
	y = y * x + coeffs[0];
	y
}
