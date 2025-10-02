pub enum TensorBufferContainer<'a, T> {
	Vec(Vec<T>),
	Slice(&'a mut [T]),
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

pub type TensorFormat = usize;

pub const FLOAT_TENSOR_BUFFER_IMAGE_RBG_FORMAT: TensorFormat = 0;
pub const FLOAT_TENSOR_BUFFER_IMAGE_RBG_255_FORMAT: TensorFormat = 1;
pub const FLOAT_TENSOR_BUFFER_MIDAS_IMAGE_RBG_FORMAT: TensorFormat = 2;
pub const FLOAT_TENSOR_BUFFER_YOLO_IMAGE_RBG_FORMAT: TensorFormat = 3;
pub const FLOAT_TENSOR_BUFFER_RELATIVE_DEPTH_FORMAT: TensorFormat = 4;
pub const FLOAT_TENSOR_BUFFER_RAW_RELATIVE_DEPTH_FORMAT: TensorFormat = 5;
pub const FLOAT_TENSOR_BUFFER_METRIC_DEPTH_FORMAT: TensorFormat = 6;
pub const FLOAT_TENSOR_BUFFER_YOLO_OUTPUT_FORMAT: TensorFormat = 7;

pub struct TensorBuffer<'a, T, const FORMAT: TensorFormat> {
	container: TensorBufferContainer<'a, T>,
}
impl<'a, T, const FORMAT: usize> TensorBuffer<'a, T, FORMAT> {
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
		}
	}

	pub fn data_mut(&mut self) -> &mut [T] {
		match &mut self.container {
			TensorBufferContainer::Vec(vec) => vec,
			TensorBufferContainer::Slice(slice) => slice,
		}
	}

	pub fn iter(&self) -> std::slice::Iter<'_, T> {
		self.data().iter()
	}

	pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
		self.data_mut().iter_mut()
	}
}
