use eye_ai_core_rs::{
	CreateYoloModelInfo, FloatTensorBuffer, FloatTensorFormat, ProfilingFrame, YoloModel,
};
use tracing::Level;
use tracing_subscriber::fmt::format::Format;

#[test]
fn run_yolo_model() {
	tracing_subscriber::fmt()
		.event_format(
			Format::default()
				.compact()
				.with_target(false)
				.with_source_location(true)
				.without_time(),
		)
		.with_max_level(Level::DEBUG)
		.init();

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
		model_data: std::fs::read("../EyeAIApp/app/src/main/assets/model.tflite")
			.expect("failed to load yolo model.tflite file"),
		labels,
		npu_config: None,
	};

	let object_profiling_frame = ProfilingFrame::new("Object");

	let mut yolo_model = YoloModel::new(yolo_model_create_info, &object_profiling_frame)
		.expect("failed to create interpreter");

	let detected_objects = yolo_model
		.run_no_preprocessing(&mut input)
		.expect("failed to run inference on yolo model");

	assert_eq!(detected_objects.len(), 1);
	assert_eq!(detected_objects[0].class_name, "cat");
}
