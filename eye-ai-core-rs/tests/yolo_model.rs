use eye_ai_core_rs::{
	CreateYoloModelInfo, FLOAT_TENSOR_BUFFER_YOLO_IMAGE_RGB_FORMAT, FloatTensorBuffer, YoloModel,
};
use std::{
	ffi::{CStr, CString},
	path::PathBuf,
};

#[test]
fn run_midas_depth_model() {
	let input_image = image::ImageReader::open("tests/cat.jpg")
		.unwrap()
		.decode()
		.unwrap()
		.resize_exact(640, 640, image::imageops::FilterType::Nearest);
	let mut input_image_rgb32f = input_image.into_rgb32f();
	let mut input_image_rgb32f = input_image_rgb32f.as_flat_samples_mut();
	let input = FloatTensorBuffer::<FLOAT_TENSOR_BUFFER_YOLO_IMAGE_RGB_FORMAT>::new(
		input_image_rgb32f.as_mut_slice(),
	);

	let tmp_dir = std::env::temp_dir().join("eye-ai-core-rs-testing-serialization-cache");
	std::fs::create_dir_all(&tmp_dir)
		.expect("failed to create tmp cache directory for delegate serialization");
	let tmp_dir_str = CString::new(tmp_dir.to_string_lossy().into_owned()).unwrap();

	let model_token = CString::new("midas_model_token").unwrap();

	let labels = std::fs::read_to_string("../EyeAIApp/app/src/main/assets/coco.names")
		.expect("failed to load labels file coco.names")
		.split('\n')
		.map(str::to_string)
		.collect::<Vec<_>>();

	let yolo_model_create_info = CreateYoloModelInfo {
		tflite_lib_filepath: PathBuf::from("libtensorflow-lite.so"),
		tflite_gpu_delegate_lib_filepath: None,
		model_data: std::fs::read("../EyeAIApp/app/src/main/assets/model.tflite")
			.expect("failed to load yolo model.tflite file"),
		labels,
		gpu_delegate_serialization_dir: tmp_dir_str.as_c_str(),
		model_token: model_token.as_c_str(),
		log_warning_callback: |message| println!("[WARN] {}", message),
		log_error_callback: |message| unsafe {
			match CStr::from_ptr(message).to_str() {
				Ok(msg) => println!("[ERROR] {}", msg),
				Err(_) => println!(
					"[ERROR] Failed to convert CString to String in order to log tflite error message"
				),
			}
		},
		npu_config: None,
	};

	let mut yolo_model =
		YoloModel::new(yolo_model_create_info).expect("failed to create interpreter");

	let detected_objects = yolo_model
		.run(input)
		.expect("failed to run inference on yolo model");

	assert_eq!(detected_objects.len(), 1);
	assert_eq!(detected_objects[0].class_name, "cat");
}
