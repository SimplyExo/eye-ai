use std::time::{Duration, Instant};

use eye_ai_core_rs::{
	BoundingBox, DetectedObject, TrackedObject,
	audio::{SpatialAudioSettings, Vec2},
};
use eye_ai_core_rs_native_lib::{
	LogCallbacks, UniffiFloatBufferWrapper, sendAIDataForSpatialAudio, setAudioSettings,
	setupAudioContent,
};

#[derive(Debug, Copy, Clone)]
struct Logger {}
impl LogCallbacks for Logger {
	fn log_info_callback(&self, msg: String) {
		println!("[INFO]  {}", msg);
	}
	fn log_warning_callback(&self, msg: String) {
		println!("[WARN]  {}", msg);
	}
	fn log_error_callback(&self, msg: String) {
		eprintln!("[ERROR] {}", msg);
	}
}

fn main() {
	let coco_labels_json_content =
		std::fs::read_to_string("../../EyeAIApp/app/src/main/assets/coco_labels_data_english.json")
			.expect("failed to read coco labels json file");
	let coco_labels_audio_file_content =
		std::fs::read("../../EyeAIApp/app/src/main/assets/coco_labels_english.wav")
			.expect("failed to read coco labels audio file");

	let logger = Logger {};

	setupAudioContent(
		coco_labels_audio_file_content,
		coco_labels_json_content,
		Box::new(logger),
	);

	setAudioSettings(SpatialAudioSettings::DEFAULT_FREQUENCY, 5, Box::new(logger));

	// TODO: replace dummy depth data and object detection by running MiDaS and YOLO off a sample video feed.

	let mut depth_estimation_data = [0.0; 256 * 256];

	let start = Instant::now();
	let mut flip_flop = false;
	loop {
		let elapsed = start.elapsed();
		if elapsed.as_secs_f32() > 10.0 {
			break;
		}

		let center = Vec2::new((elapsed.as_secs_f32() * 2.0).sin() * 0.5 + 0.5, 0.5);

		// dummy object detections
		let cat_bbox = BoundingBox {
			center_x: center.x,
			center_y: center.y,
			width: 0.25,
			height: 0.4,
		};
		let tracked_object = TrackedObject {
			object: DetectedObject {
				class_name: if flip_flop {
					"cat".to_string()
				} else {
					"dog".to_string()
				},
				class_id: if flip_flop {
					15 // line 16 "cat" in coco.names
				} else {
					16 // line 17 "dog" in coco.names
				},
				confidence: 0.7,
				bbox: cat_bbox.clone(),
			},
			tracking_id: 0,
		};

		// dummy depth data
		let cat_distance = 0.5;
		let background_distance = 5.0;
		for y in 0..256 {
			for x in 0..256 {
				let index = y * 256 + x;
				let relative_x = x as f32 / 255.0;
				let relative_y = y as f32 / 255.0;

				depth_estimation_data[index] =
					if cat_bbox.contains(Vec2::new(relative_x, relative_y)) {
						cat_distance
					} else {
						background_distance
					};
			}
		}

		sendAIDataForSpatialAudio(
			UniffiFloatBufferWrapper {
				ptr_address: depth_estimation_data.as_ptr() as i64,
				length: depth_estimation_data.len() as i32,
			},
			vec![tracked_object.into()],
			Box::new(logger),
		);

		std::thread::sleep(Duration::from_millis(100));

		flip_flop = !flip_flop;
	}
}
