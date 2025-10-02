pub mod tflite;

mod tensor_buffer;
pub use tensor_buffer::{
	FLOAT_TENSOR_BUFFER_MIDAS_IMAGE_RBG_FORMAT, FLOAT_TENSOR_BUFFER_RAW_RELATIVE_DEPTH_FORMAT,
	TensorBuffer, TensorFormat,
};

pub fn greet() -> String {
	"Hello from eye-ai-core-rs! Getting called by eye-ai-core-rs-native-lib".to_string()
}
