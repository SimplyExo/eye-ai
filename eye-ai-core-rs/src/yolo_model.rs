use crate::{
	FLOAT_TENSOR_BUFFER_YOLO_IMAGE_RGB_FORMAT, FLOAT_TENSOR_BUFFER_YOLO_OUTPUT_FORMAT,
	FloatTensorBuffer,
	tflite::{
		CreateTfLiteRuntimeInfo, NpuConfig, NpuConfigType, TfLiteRunInferenceError, TfLiteRuntime,
		TfLiteRuntimeCreateError,
	},
};
use std::path::PathBuf;

pub struct YoloModelNpuConfig<'a> {
	pub tflite_qnn_npu_delegate_lib_filepath: PathBuf,
	pub skel_library_dir: &'a std::ffi::CStr,
}

pub struct CreateYoloModelInfo<'a> {
	pub labels: Vec<String>,
	pub tflite_lib_filepath: PathBuf,
	/// if None, we try to load gpu delegate api from the tflite_lib_filepath library
	pub tflite_gpu_delegate_lib_filepath: Option<PathBuf>,
	pub model_data: Vec<u8>,
	pub gpu_delegate_serialization_dir: &'a std::ffi::CStr,
	pub model_token: &'a std::ffi::CStr,
	pub log_warning_callback: fn(msg: &str),
	pub log_error_callback: fn(msg: *const std::os::raw::c_char),
	pub npu_config: Option<YoloModelNpuConfig<'a>>,
}

pub struct YoloModel {
	runtime: TfLiteRuntime,
	labels: Vec<String>,
	num_elements: usize,
	num_channel: usize,
}

#[derive(Debug, Clone)]
pub struct BoundingBox {
	pub center_x: f32,
	pub center_y: f32,
	pub width: f32,
	pub height: f32,
}
impl BoundingBox {
	pub fn x1(&self) -> f32 {
		self.center_x - self.width / 2.0
	}
	pub fn x2(&self) -> f32 {
		self.center_x + self.width / 2.0
	}
	pub fn y1(&self) -> f32 {
		self.center_y - self.height / 2.0
	}
	pub fn y2(&self) -> f32 {
		self.center_y + self.height / 2.0
	}
	pub fn is_valid(&self) -> bool {
		let valid_range = 0.0..1.0;
		valid_range.contains(&self.x1())
			&& valid_range.contains(&self.x2())
			&& valid_range.contains(&self.y1())
			&& valid_range.contains(&self.y2())
	}
}

#[derive(Debug, Clone)]
pub struct DetectedObject {
	pub class_name: String,
	pub class_id: usize,
	pub confidence: f32,
	pub bbox: BoundingBox,
}

impl YoloModel {
	const CONFIDENCE_THRESHOLD: f32 = 0.5;
	const IOU_THRESHOLD: f32 = 0.5;

	pub fn new(create_info: CreateYoloModelInfo) -> Result<Self, TfLiteRuntimeCreateError> {
		let runtime_create_info = CreateTfLiteRuntimeInfo {
			tflite_lib_filepath: create_info.tflite_lib_filepath,
			tflite_gpu_delegate_lib_filepath: create_info.tflite_gpu_delegate_lib_filepath,
			model_data: create_info.model_data,
			gpu_delegate_serialization_dir: create_info.gpu_delegate_serialization_dir,
			model_token: create_info.model_token,
			model_input_format: FLOAT_TENSOR_BUFFER_YOLO_IMAGE_RGB_FORMAT,
			model_output_format: FLOAT_TENSOR_BUFFER_YOLO_OUTPUT_FORMAT,
			log_warning_callback: create_info.log_warning_callback,
			log_error_callback: create_info.log_error_callback,
			npu_config: create_info.npu_config.map(|depth_npu_config| NpuConfig {
				tflite_qnn_npu_delegate_lib_filepath: depth_npu_config
					.tflite_qnn_npu_delegate_lib_filepath,
				skel_library_dir: depth_npu_config.skel_library_dir,
				config_type: NpuConfigType::Yolo,
			}),
		};

		let runtime = TfLiteRuntime::new(runtime_create_info)?;

		let output_shape =
			runtime
				.get_output_shape()
				.ok_or(TfLiteRuntimeCreateError::TfLiteDyn(
					tflite_dyn::Error::Generic,
				))?;

		let num_channel = output_shape[1] as usize;
		let num_elements = output_shape[2] as usize;

		Ok(Self {
			runtime,
			labels: create_info.labels,
			num_channel,
			num_elements,
		})
	}

	pub fn run(
		&mut self,
		input_tensor: FloatTensorBuffer<FLOAT_TENSOR_BUFFER_YOLO_IMAGE_RGB_FORMAT>,
	) -> Result<Vec<DetectedObject>, TfLiteRunInferenceError> {
		let output_tensor: FloatTensorBuffer<FLOAT_TENSOR_BUFFER_YOLO_OUTPUT_FORMAT> =
			self.runtime.run_inference_with_tensors(input_tensor)?;

		Ok(best_objects(
			output_tensor.data(),
			&self.labels,
			self.num_elements,
			self.num_channel,
			Self::CONFIDENCE_THRESHOLD,
			Self::IOU_THRESHOLD,
		))
	}
}

fn best_objects(
	array: &[f32],
	labels: &[String],
	num_elements: usize,
	num_channel: usize,
	confidence_threshold: f32,
	iou_threshold: f32,
) -> Vec<DetectedObject> {
	let mut objects = Vec::new();
	let actual_size = num_elements * num_channel;

	if array.len() < actual_size {
		return Vec::new();
	}

	for c in 0..num_elements {
		if let Some(object) = parse_object(
			array,
			labels,
			c,
			num_elements,
			num_channel,
			confidence_threshold,
		) {
			objects.push(object);
		}
	}

	apply_nms(&objects, iou_threshold)
}

fn parse_object(
	array: &[f32],
	labels: &[String],
	box_index: usize,
	num_elements: usize,
	num_channel: usize,
	confidence_threshold: f32,
) -> Option<DetectedObject> {
	let mut max_confidence = -1.0;
	let mut max_index = -1;
	let mut array_index = box_index + (num_elements * 4);

	for i in 4..num_channel {
		if array_index >= array.len() {
			break;
		}

		let confidence = array[array_index];
		if confidence > max_confidence {
			max_confidence = confidence;
			max_index = (i - 4) as i32;
		}

		array_index += num_elements;
	}

	if max_confidence < confidence_threshold {
		return None;
	}

	if max_index < 0 || max_index >= labels.len() as i32 {
		return None;
	}

	let bbox = BoundingBox {
		center_x: array[box_index],
		center_y: array[box_index + num_elements],
		width: array[box_index + (num_elements * 2)],
		height: array[box_index + (num_elements * 3)],
	};
	if !bbox.is_valid() {
		return None;
	}

	let class_name = labels[max_index as usize].clone();
	Some(DetectedObject {
		class_name,
		class_id: max_index as usize,
		bbox,
		confidence: max_confidence,
	})
}

fn apply_nms(objects: &[DetectedObject], iou_threshold: f32) -> Vec<DetectedObject> {
	if objects.is_empty() {
		return Vec::new();
	}

	// sorted descending
	let mut sorted_objects = objects.to_vec();
	sorted_objects.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));

	let mut selected_objects = Vec::new();

	// TODO: pop elements in reverse instead -> not so much reallocations
	while let Some(first) = sorted_objects.first().cloned() {
		selected_objects.push(first.clone());
		sorted_objects.remove(0);

		let mut i = 0;
		while i < sorted_objects.len() {
			let iou = calculate_iou(&first, &sorted_objects[i]);
			if iou >= iou_threshold {
				sorted_objects.remove(i);
			} else {
				i += 1;
			}
		}
	}

	selected_objects
}

fn calculate_iou(object_a: &DetectedObject, object_b: &DetectedObject) -> f32 {
	let x1 = f32::max(object_a.bbox.x1(), object_b.bbox.x1());
	let y1 = f32::max(object_a.bbox.y1(), object_b.bbox.y1());
	let x2 = f32::max(object_a.bbox.x2(), object_b.bbox.x2());
	let y2 = f32::max(object_a.bbox.y2(), object_b.bbox.y2());

	let intersection_width = f32::max(0.0, x2 - x1);
	let intersection_height = f32::max(0.0, y2 - y1);
	let intersection_area = intersection_width * intersection_height;

	let a_area = object_a.bbox.width * object_a.bbox.height;
	let b_area = object_b.bbox.width * object_b.bbox.height;

	intersection_area / (a_area + b_area - intersection_area)
}
