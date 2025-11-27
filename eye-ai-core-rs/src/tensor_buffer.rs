use thiserror::Error;

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
	YoloOutput,
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
				Self::YoloOutput => "YOLO Output",
			}
		)
	}
}

pub type FloatTensorBuffer<'a> = TensorBuffer<'a, f32, FloatTensorFormat>;

#[derive(Debug, Clone, Copy, Error)]
pub struct WrongFloatTensorFormatError {
	expected: FloatTensorFormat,
	given: FloatTensorFormat,
	tensor_name: &'static str,
}
impl std::fmt::Display for WrongFloatTensorFormatError {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(
			f,
			"tensor {} needs to be {}, but was provided as {}",
			self.tensor_name, self.expected, self.given
		)
	}
}

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

pub fn check_float_tensor_format(
	tensor: &FloatTensorBuffer,
	tensor_name: &'static str,
	expected_format: FloatTensorFormat,
) -> Result<(), WrongFloatTensorFormatError> {
	if tensor.format() == expected_format {
		Ok(())
	} else {
		Err(WrongFloatTensorFormatError {
			expected: expected_format,
			given: tensor.format(),
			tensor_name,
		})
	}
}

#[macro_export]
macro_rules! check_float_tensor_format {
	($tensor:expr,$expected_format:expr) => {{
		check_float_tensor_format($tensor, stringify!($tensor), $expected_format)?;
	}};
}

/// takes FloatTensorFormat::ImageRgb255, returns FloatTensorFormat::MiDaSImageRgb
pub fn image_rgb_255_to_midas_image<'a>(
	image_rgb_tensor: &mut FloatTensorBuffer<'a>,
) -> Result<(), WrongFloatTensorFormatError> {
	check_float_tensor_format!(image_rgb_tensor, FloatTensorFormat::ImageRgb255);

	let mean = [123.675, 116.28, 103.53];
	let std = [58.395, 57.12, 57.375];

	assert_eq!(image_rgb_tensor.data().len() % 3, 0);

	let values = image_rgb_tensor.data_mut();

	for i in 0..(values.len() / 3) {
		values[3 * i] = (values[3 * i] - mean[0]) / std[0];
		values[3 * i + 1] = (values[3 * i + 1] - mean[1]) / std[1];
		values[3 * i + 2] = (values[3 * i + 2] - mean[2]) / std[2];
	}
	image_rgb_tensor.convert_format(FloatTensorFormat::MiDaSImageRgb);

	Ok(())
}
