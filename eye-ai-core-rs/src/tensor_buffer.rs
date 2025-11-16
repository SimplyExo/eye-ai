#[derive(Debug)]
pub enum TensorBufferContainer<'a, T> {
	Vec(Vec<T>),
	Slice(&'a [T]),
	MutSlice(&'a mut [T]),
}
impl<'a, T: Clone> Clone for TensorBufferContainer<'a, T> {
	fn clone(&self) -> Self {
		match self {
			Self::Vec(vec) => Self::Vec(vec.clone()),
			Self::Slice(slice) => Self::Vec(slice.to_vec()),
			Self::MutSlice(slice) => Self::Vec(slice.to_vec()),
		}
	}
}
impl<'a, T> From<Vec<T>> for TensorBufferContainer<'a, T> {
	fn from(value: Vec<T>) -> Self {
		TensorBufferContainer::Vec(value)
	}
}
impl<'a, T> From<&'a [T]> for TensorBufferContainer<'a, T> {
	fn from(value: &'a [T]) -> Self {
		TensorBufferContainer::Slice(value)
	}
}
impl<'a, T, const N: usize> From<&'a [T; N]> for TensorBufferContainer<'a, T> {
	fn from(value: &'a [T; N]) -> Self {
		TensorBufferContainer::Slice(value)
	}
}
impl<'a, T> From<&'a mut [T]> for TensorBufferContainer<'a, T> {
	fn from(value: &'a mut [T]) -> Self {
		TensorBufferContainer::MutSlice(value)
	}
}
impl<'a, T, const N: usize> From<&'a mut [T; N]> for TensorBufferContainer<'a, T> {
	fn from(value: &'a mut [T; N]) -> Self {
		TensorBufferContainer::MutSlice(value)
	}
}

pub type TensorFormat = usize;

#[allow(unused)]
pub const FLOAT_TENSOR_BUFFER_IMAGE_RGB_FORMAT: TensorFormat = 0;
#[allow(unused)]
pub const FLOAT_TENSOR_BUFFER_IMAGE_RGB_255_FORMAT: TensorFormat = 1;
pub const FLOAT_TENSOR_BUFFER_MIDAS_IMAGE_RGB_FORMAT: TensorFormat = 2;
#[allow(unused)]
pub const FLOAT_TENSOR_BUFFER_YOLO_IMAGE_RGB_FORMAT: TensorFormat = 3;
#[allow(unused)]
pub const FLOAT_TENSOR_BUFFER_RELATIVE_DEPTH_FORMAT: TensorFormat = 4;
pub const FLOAT_TENSOR_BUFFER_RAW_RELATIVE_DEPTH_FORMAT: TensorFormat = 5;
#[allow(unused)]
pub const FLOAT_TENSOR_BUFFER_METRIC_DEPTH_FORMAT: TensorFormat = 6;
#[allow(unused)]
pub const FLOAT_TENSOR_BUFFER_YOLO_OUTPUT_FORMAT: TensorFormat = 7;

pub fn get_tensor_format_name(format: TensorFormat) -> &'static str {
	match format {
		FLOAT_TENSOR_BUFFER_IMAGE_RGB_FORMAT => "Float Image RGB",
		FLOAT_TENSOR_BUFFER_IMAGE_RGB_255_FORMAT => "Float Image RGB 255",
		FLOAT_TENSOR_BUFFER_MIDAS_IMAGE_RGB_FORMAT => "Float Image RGB Midas",
		FLOAT_TENSOR_BUFFER_RAW_RELATIVE_DEPTH_FORMAT => "Float Raw Relative Depth",
		FLOAT_TENSOR_BUFFER_RELATIVE_DEPTH_FORMAT => "Float Relative Depth",
		FLOAT_TENSOR_BUFFER_METRIC_DEPTH_FORMAT => "Float Metric Depth",
		FLOAT_TENSOR_BUFFER_YOLO_OUTPUT_FORMAT => "Float YOLO Output",
		_ => "UNKNOWN_FORMAT",
	}
}

pub type FloatTensorBuffer<'a, const FORMAT: TensorFormat> = TensorBuffer<'a, f32, FORMAT>;

#[derive(Debug, Clone)]
pub struct TensorBuffer<'a, T, const FORMAT: TensorFormat> {
	container: TensorBufferContainer<'a, T>,
}
impl<'a, T: Clone, const FORMAT: usize> TensorBuffer<'a, T, FORMAT> {
	pub fn new(container: impl Into<TensorBufferContainer<'a, T>>) -> Self {
		Self {
			container: container.into(),
		}
	}

	pub fn convert_format<const NEW_FORMAT: usize>(self) -> TensorBuffer<'a, T, NEW_FORMAT> {
		TensorBuffer {
			container: self.container,
		}
	}

	pub fn data(&self) -> &[T] {
		match &self.container {
			TensorBufferContainer::Vec(vec) => vec,
			TensorBufferContainer::Slice(slice) => slice,
			TensorBufferContainer::MutSlice(slice) => slice,
		}
	}

	/// performance edge case: this call will clone the non mutable slice to a mutable vec
	pub fn data_mut(&mut self) -> &mut [T] {
		if let TensorBufferContainer::Slice(slice) = &self.container {
			self.container = TensorBufferContainer::Vec(slice.to_vec());
		}

		match &mut self.container {
			TensorBufferContainer::Vec(vec) => vec,
			TensorBufferContainer::Slice(_slice) => {
				unreachable!("should have been converted to a TensorBufferContainer::Vec!!!")
			}
			TensorBufferContainer::MutSlice(slice) => slice,
		}
	}

	pub fn to_vec(self) -> Vec<T> {
		match self.container {
			TensorBufferContainer::Vec(vec) => vec,
			TensorBufferContainer::Slice(slice) => slice.to_vec(),
			TensorBufferContainer::MutSlice(slice) => slice.to_vec(),
		}
	}

	pub fn iter(&self) -> std::slice::Iter<'_, T> {
		self.data().iter()
	}

	pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
		self.data_mut().iter_mut()
	}
}

impl<'a> From<FloatTensorBuffer<'a, FLOAT_TENSOR_BUFFER_IMAGE_RGB_255_FORMAT>>
	for FloatTensorBuffer<'a, FLOAT_TENSOR_BUFFER_MIDAS_IMAGE_RGB_FORMAT>
{
	fn from(
		mut image_rgb_tensor: FloatTensorBuffer<'a, FLOAT_TENSOR_BUFFER_IMAGE_RGB_255_FORMAT>,
	) -> Self {
		let mean = [123.675, 116.28, 103.53];
		let std = [58.395, 57.12, 57.375];

		assert_eq!(image_rgb_tensor.data().len() % 3, 0);

		let values = image_rgb_tensor.data_mut();

		for i in 0..(values.len() / 3) {
			values[3 * i] = (values[3 * i] - mean[0]) / std[0];
			values[3 * i + 1] = (values[3 * i + 1] - mean[1]) / std[1];
			values[3 * i + 2] = (values[3 * i + 2] - mean[2]) / std[2];
		}
		image_rgb_tensor.convert_format()
	}
}
