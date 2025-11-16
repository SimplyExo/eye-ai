use eye_ai_core_rs::{
	BoundingBox, DetectedObject, ProfilingFrame,
	audio::{SpatialAudio, SpatialAudioSettings, Vec2},
};
use std::{
	sync::{Arc, RwLock},
	time::{Duration, Instant},
};

fn main() {
	let audio_profiling_frame = Arc::new(ProfilingFrame::new("Audio"));

	let coco_labels_json_content =
		std::fs::read_to_string("../EyeAIApp/app/src/main/assets/coco_labels_data_english.json")
			.unwrap();
	let coco_labels_audio_file_content =
		std::fs::read("../EyeAIApp/app/src/main/assets/coco_labels_english.wav")
			.expect("failed to read coco labels audio file");
	let log_info_callback = |message: &str| println!("INFO: {}", message);
	let log_error_callback = |message: &str| eprintln!("ERROR: {}", message);
	let spatial_audio_settings = SpatialAudioSettings::new(
		coco_labels_audio_file_content,
		coco_labels_json_content,
		log_info_callback,
		log_error_callback,
	);
	let mut spatial_audio = SpatialAudio::new(
		Arc::new(RwLock::new(spatial_audio_settings)),
		audio_profiling_frame,
	)
	.unwrap();

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
		let object_detection_data = vec![DetectedObject {
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
		}];

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

		spatial_audio.update(&depth_estimation_data, &object_detection_data);

		std::thread::sleep(Duration::from_millis(100));

		flip_flop = !flip_flop;
	}
}
