use eye_ai_core_rs::{
	CreateYoloModelInfo, DetectedObject, FloatTensorBuffer, FloatTensorFormat, ProfilingFrame,
	YoloModel,
};
use tracing::Level;
use tracing_subscriber::fmt::format::Format;

fn run_yolo_model(model_filename: &str) -> Vec<DetectedObject> {
	let _ = tracing_subscriber::fmt()
		.event_format(
			Format::default()
				.compact()
				.with_target(false)
				.with_source_location(true)
				.without_time(),
		)
		.with_max_level(Level::DEBUG)
		.try_init();

	#[cfg(feature = "enable_tracy_profiling")]
	tracing_tracy::client::Client::start();

	let input_image = image::ImageReader::open("tests/cat.jpg")
		.unwrap()
		.decode()
		.unwrap()
		.resize_exact(640, 640, image::imageops::FilterType::Nearest);
	let mut input_image_rgb32f = input_image.into_rgb32f();
	let mut input_image_rgb32f = input_image_rgb32f.as_flat_samples_mut();
	let mut input = FloatTensorBuffer::new(
		input_image_rgb32f.as_mut_slice(),
		FloatTensorFormat::YoloImageRgb,
	);

	let labels = std::fs::read_to_string("../EyeAIApp/app/src/main/assets/coco.names")
		.expect("failed to load labels file coco.names")
		.split('\n')
		.map(str::to_string)
		.collect::<Vec<_>>();

	let yolo_model_create_info = CreateYoloModelInfo {
		model_name: "YOLO".to_string(),
		model_data: std::fs::read(format!("../EyeAIApp/app/src/main/assets/{model_filename}"))
			.expect("failed to load yolo model file"),
		labels,
		npu_config: None,
	};

	let object_profiling_frame = ProfilingFrame::new("Object");

	let mut yolo_model = YoloModel::new(yolo_model_create_info, &object_profiling_frame)
		.expect("failed to create interpreter");

	yolo_model
		.run_no_preprocessing(&mut input)
		.expect("failed to run inference on yolo model")
}

#[test]
fn run_float_yolo_model() {
	let detected_objects = run_yolo_model("model.tflite");

	assert_eq!(detected_objects.len(), 1);
	assert_eq!(detected_objects[0].class_name, "cat");
}

#[test]
fn run_quantized_yolo_model() {
	let detected_objects = run_yolo_model("yolov8n_int8.tflite");
	let cat = detected_objects
		.iter()
		.find(|object| object.class_name == "cat")
		.expect("quantized YOLO model did not detect the cat");

	assert!(cat.confidence >= 0.5);
	assert!(cat.bbox.is_valid());
}
