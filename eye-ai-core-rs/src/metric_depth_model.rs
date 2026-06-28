use eye_ai_core_rs_profiling_attribute::profile_function;

use crate::{
	CreateDepthModelInfo, DepthModel, FloatTensorBuffer, FloatTensorFormat, ProfilingFrame,
	check_float_tensor_format,
	tensor_buffer::WrongFloatTensorFormatError,
	tflite::{LiteRtRunInferenceError, LiteRtRuntimeCreateError},
};

#[derive(Debug)]
pub struct MetricDepthModel<'a> {
	relative_depth_model: DepthModel<'a>,
	profiling_frame: &'a ProfilingFrame,
}
impl<'a> MetricDepthModel<'a> {
	const REL2ABS_COEFFS: [f32; 5] = [4.30595, -6.5995E-03, 5.25059E-6, -2.7962E-9, 9.28594E-13];

	pub fn new(
		relative_depth_model_create_info: CreateDepthModelInfo,
		profiling_frame: &'a ProfilingFrame,
	) -> Result<Self, LiteRtRuntimeCreateError> {
		let relative_depth_model =
			DepthModel::new(relative_depth_model_create_info, profiling_frame)?;

		Ok(Self {
			relative_depth_model,
			profiling_frame,
		})
	}

	/// input format: FloatTensorFormat::MiDaSImageRgb, output format will be: FloatTensorFormat::MetricDepth
	#[profile_function("self.profiling_frame")]
	pub fn run(
		&mut self,
		input_tensor: &mut FloatTensorBuffer,
		output_tensor: &mut FloatTensorBuffer,
	) -> Result<(), LiteRtRunInferenceError> {
		check_float_tensor_format!(input_tensor, FloatTensorFormat::MiDaSImageRgb);

		self.relative_depth_model
			.run_raw(input_tensor, output_tensor)?;

		rel2abs_operator(output_tensor, &Self::REL2ABS_COEFFS, self.profiling_frame)?;

		check_float_tensor_format!(output_tensor, FloatTensorFormat::MetricDepth);

		Ok(())
	}

	pub fn get_input_shape(&self) -> Option<Vec<i32>> {
		self.relative_depth_model.get_input_shape()
	}

	pub fn get_output_shape(&self) -> Option<Vec<i32>> {
		self.relative_depth_model.get_output_shape()
	}
}

/// converts FloatTensorFormat::RawRelativeDepth to FloatTensorFormat::MetricDepth
#[profile_function("profiling_frame")]
fn rel2abs_operator<'a>(
	raw_relative_depth_tensor: &mut FloatTensorBuffer<'a>,
	rel2abs_coeffs: &'a [f32; 5],
	profiling_frame: &ProfilingFrame,
) -> Result<(), WrongFloatTensorFormatError> {
	check_float_tensor_format!(
		raw_relative_depth_tensor,
		FloatTensorFormat::RawRelativeDepth
	);

	raw_relative_depth_tensor.convert_format(FloatTensorFormat::MetricDepth);

	for value in raw_relative_depth_tensor.iter_mut() {
		*value = polynomial_n4(*value, rel2abs_coeffs);
	}

	Ok(())
}

/// Polynomial function of degree 4 using Horner's method.
///
/// coeffs: {a0, a1, a2, a3, a4}
///
/// polynomial_4(x) = a0 + a1 * x + a2 * x² + a3 * x³ + a4 * x⁴
const fn polynomial_n4(x: f32, coeffs: &[f32; 5]) -> f32 {
	let mut y = coeffs[4];
	y = y * x + coeffs[3];
	y = y * x + coeffs[2];
	y = y * x + coeffs[1];
	y = y * x + coeffs[0];
	y
}

#[allow(unused)]
fn polynomial_n<const N: usize>(x: f32, coeffs: &[f32; N]) -> f32 {
	let mut y = coeffs[N - 1];
	for i in 0..(N - 2) {
		y = y * x + coeffs[(N - 2) - i];
	}
	y
}
