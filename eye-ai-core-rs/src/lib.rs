pub mod tflite;

mod tensor_buffer;
pub use tensor_buffer::{
	FLOAT_TENSOR_BUFFER_IMAGE_RGB_255_FORMAT, FLOAT_TENSOR_BUFFER_IMAGE_RGB_FORMAT,
	FLOAT_TENSOR_BUFFER_METRIC_DEPTH_FORMAT, FLOAT_TENSOR_BUFFER_MIDAS_IMAGE_RGB_FORMAT,
	FLOAT_TENSOR_BUFFER_RAW_RELATIVE_DEPTH_FORMAT, FLOAT_TENSOR_BUFFER_RELATIVE_DEPTH_FORMAT,
	FLOAT_TENSOR_BUFFER_YOLO_IMAGE_RGB_FORMAT, FLOAT_TENSOR_BUFFER_YOLO_OUTPUT_FORMAT,
	FloatTensorBuffer, TensorBuffer, TensorBufferContainer, TensorFormat, get_tensor_format_name,
};

mod depth_model;
pub use depth_model::{CreateDepthModelInfo, DepthModel, DepthModelNpuConfig};

mod metric_depth_model;
pub use metric_depth_model::MetricDepthModel;

pub fn greet() -> String {
	"Hello from eye-ai-core-rs! Getting called by eye-ai-core-rs-native-lib".to_string()
}
