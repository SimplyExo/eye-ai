use std::simd::f32x8;

#[derive(Debug)]
pub enum TensorBufferContainer<'a, T> {
	Vec(Vec<T>),
	Slice(&'a mut [T]),
}

impl<'a, T: Clone> Clone for TensorBufferContainer<'a, T> {
	fn clone(&self) -> Self {
		match self {
			Self::Vec(vec) => Self::Vec(vec.clone()),
			Self::Slice(slice) => Self::Vec(slice.to_vec()),
		}
	}
}
impl<'a, T> From<Vec<T>> for TensorBufferContainer<'a, T> {
	fn from(value: Vec<T>) -> Self {
		TensorBufferContainer::Vec(value)
	}
}
impl<'a, T> From<&'a mut [T]> for TensorBufferContainer<'a, T> {
	fn from(value: &'a mut [T]) -> Self {
		TensorBufferContainer::Slice(value)
	}
}
impl<'a, T, const N: usize> From<&'a mut [T; N]> for TensorBufferContainer<'a, T> {
	fn from(value: &'a mut [T; N]) -> Self {
		TensorBufferContainer::Slice(value)
	}
}

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum FloatTensorFormat {
	ImageRgb,
	ImageRgb255,
	MiDaSImageRgb,
	YoloImageRgb,
	RelativeDepth,
	RawRelativeDepth,
	MetricDepth,
	YoloObjectDetectionOutput,
	YoloSegmentationOutput,
}
impl std::fmt::Display for FloatTensorFormat {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(
			f,
			"{}",
			match self {
				Self::ImageRgb => "Image RGB",
				Self::ImageRgb255 => "Image RGB 255",
				Self::MiDaSImageRgb => "Image RGB MiDaS",
				Self::YoloImageRgb => "Image RGB YOLO",
				Self::RawRelativeDepth => "Raw Relative Depth",
				Self::RelativeDepth => "Relative Depth",
				Self::MetricDepth => "Metric Depth",
				Self::YoloObjectDetectionOutput => "YOLO Object Detection Output",
				Self::YoloSegmentationOutput => "YOLO Segmentation Output",
			}
		)
	}
}

pub type FloatTensorBuffer<'a> = TensorBuffer<'a, f32, FloatTensorFormat>;

#[derive(Debug)]
pub struct TensorBuffer<'a, T: Clone, Format: Eq + Copy> {
	container: TensorBufferContainer<'a, T>,
	format: Format,
}
impl<'a, T: Clone, Format: Eq + Copy> TensorBuffer<'a, T, Format> {
	pub fn new(container: impl Into<TensorBufferContainer<'a, T>>, format: Format) -> Self {
		Self {
			container: container.into(),
			format,
		}
	}

	pub fn format(&self) -> Format {
		self.format
	}

	pub fn convert_format(&mut self, new_format: Format) {
		self.format = new_format;
	}

	pub fn data(&self) -> &[T] {
		match &self.container {
			TensorBufferContainer::Slice(slice) => slice,
			TensorBufferContainer::Vec(vec) => vec,
		}
	}

	pub fn data_mut(&mut self) -> &mut [T] {
		match &mut self.container {
			TensorBufferContainer::Slice(slice) => slice,
			TensorBufferContainer::Vec(vec) => vec.as_mut_slice(),
		}
	}

	pub fn iter(&self) -> std::slice::Iter<'_, T> {
		self.data().iter()
	}

	pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
		self.data_mut().iter_mut()
	}
}

/// panics when the given tensor's format does not match the expected format
pub fn check_float_tensor_format(
	tensor: &FloatTensorBuffer,
	tensor_name: &'static str,
	expected_format: FloatTensorFormat,
) {
	let actual_format = tensor.format();
	if actual_format != expected_format {
		panic!(
			"tensor {tensor_name} needs to be in {expected_format} format, but was in {actual_format} format"
		);
	}
}

#[macro_export]
macro_rules! check_float_tensor_format {
	($tensor:expr,$expected_format:expr) => {{
		check_float_tensor_format($tensor, stringify!($tensor), $expected_format);
	}};
}

/// takes FloatTensorFormat::ImageRgb255, returns FloatTensorFormat::MiDaSImageRgb
pub fn image_rgb_255_to_midas_image<'a>(image_rgb_tensor: &mut FloatTensorBuffer<'a>) {
	check_float_tensor_format!(image_rgb_tensor, FloatTensorFormat::ImageRgb255);

	assert_eq!(image_rgb_tensor.data().len() % 3, 0);

	rgb_255_to_midas(image_rgb_tensor.data_mut());

	image_rgb_tensor.convert_format(FloatTensorFormat::MiDaSImageRgb);
}

const MIDAS_SIMD_BLOCK: usize = 24;

/// processes one full `MIDAS_SIMD_BLOCK` chunk with SIMD, or any smaller tail
/// chunk with scalar per-pixel code (dropping up to two stray values, matching
/// the historical `remainder.as_chunks_mut::<3>().0` handling)
#[inline]
fn midas_process_block(
	block: &mut [f32],
	mean_vectors: &[f32x8; 3],
	scale_vectors: &[f32x8; 3],
	mean: &[f32; 3],
	inv_std: &[f32; 3],
) {
	debug_assert!(block.len() <= MIDAS_SIMD_BLOCK);

	if block.len() == MIDAS_SIMD_BLOCK {
		for (lane, (mean, scale)) in mean_vectors.iter().zip(scale_vectors.iter()).enumerate() {
			let range = lane * 8..lane * 8 + 8;
			let values = f32x8::from_slice(&block[range.clone()]);
			let scaled = (values - *mean) * *scale;
			block[range].copy_from_slice(scaled.as_array());
		}
	} else {
		for pixel in block.as_chunks_mut::<3>().0 {
			pixel[0] = (pixel[0] - mean[0]) * inv_std[0];
			pixel[1] = (pixel[1] - mean[1]) * inv_std[1];
			pixel[2] = (pixel[2] - mean[2]) * inv_std[2];
		}
	}
}

pub(crate) fn rgb_255_to_midas(values: &mut [f32]) {
	let mean: [f32; 3] = [123.675, 116.28, 103.53];
	let inv_std: [f32; 3] = [58.395, 57.12, 57.375].map(f32::recip);

	let mut mean_vectors = [[0.0f32; 8]; 3];
	let mut scale_vectors = [[0.0f32; 8]; 3];
	for index in 0..MIDAS_SIMD_BLOCK {
		let channel = index % 3;
		mean_vectors[index / 8][index % 8] = mean[channel];
		scale_vectors[index / 8][index % 8] = inv_std[channel];
	}
	let mean_vectors = mean_vectors.map(f32x8::from_array);
	let scale_vectors = scale_vectors.map(f32x8::from_array);

	let (blocks, remainder) = values.as_chunks_mut::<MIDAS_SIMD_BLOCK>();
	for block in blocks {
		midas_process_block(block, &mean_vectors, &scale_vectors, &mean, &inv_std);
	}
	midas_process_block(remainder, &mean_vectors, &scale_vectors, &mean, &inv_std);
}
