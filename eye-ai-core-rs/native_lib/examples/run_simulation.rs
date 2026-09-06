use std::time::{Duration, Instant};

use eye_ai_core_rs::{
	BoundingBox, DetectedObject, TrackedObject,
	audio::{SpatialAudioSettings, Vec2},
};
use eye_ai_core_rs_native_lib::{
	UniffiDetectedObject, beginSpatialAudioSession, destroySpatialAudio, sendAIDataForSpatialAudio,
	setAudioSettings, setupAudioContent,
};
use tracing::{info, level_filters::LevelFilter};
use tracing_subscriber::{Layer, fmt::format::Format, layer::SubscriberExt};
#[cfg(feature = "enable_tracy_profiling")]
use tracing_tracy::TracyLayer;

fn main() {
	let stdout_tracing_layer = tracing_subscriber::fmt::layer()
		.event_format(
			Format::default()
				.compact()
				.with_target(true)
				.without_time()
				.with_source_location(false),
		)
		.with_filter(LevelFilter::INFO);

	#[cfg(feature = "enable_tracy_profiling")]
	let tracing_registry = tracing_subscriber::registry()
		.with(TracyLayer::default().with_filter(LevelFilter::TRACE))
		.with(stdout_tracing_layer);

	#[cfg(not(feature = "enable_tracy_profiling"))]
	let tracing_registry = tracing_subscriber::registry().with(stdout_tracing_layer);

	tracing::subscriber::set_global_default(tracing_registry)
		.expect("failed to set global tracing registry");

	#[cfg(feature = "enable_tracy_profiling")]
	tracing_tracy::client::Client::start();

	let coco_labels_json_content =
		std::fs::read_to_string("../../EyeAIApp/app/src/main/assets/coco_labels_data_english.json")
			.expect("failed to read coco labels json file");
	let coco_labels_audio_file_content =
		std::fs::read("../../EyeAIApp/app/src/main/assets/coco_labels_english.wav")
			.expect("failed to read coco labels audio file");

	setupAudioContent(coco_labels_audio_file_content, coco_labels_json_content);

	let session = beginSpatialAudioSession();
	setAudioSettings(session, SpatialAudioSettings::DEFAULT_FREQUENCY, 5);

	// TODO: replace dummy depth data and object detection by running MiDaS and YOLO off a sample video feed.

	let mut depth_estimation_data = [0.0; 256 * 256];
	let mut tracked_objects: Vec<UniffiDetectedObject> = Vec::new();

	let start = Instant::now();
	let mut flip_flop = false;
	loop {
		let elapsed = start.elapsed();
		if elapsed.as_secs_f32() > 10.0 {
			info!("Stopping Simulation, 10s elapsed");
			break;
		}

		{
			tracing::info_span!("create dummy depth and object data").in_scope(|| {
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
				tracked_objects = vec![tracked_object.into()];

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
			});
		}

		sendAIDataForSpatialAudio(
			session,
			(&mut depth_estimation_data).into(),
			tracked_objects.clone(),
		);

		{
			tracing::info_span!("sleep 100ms").in_scope(|| {
				std::thread::sleep(Duration::from_millis(100));
			});
		}

		flip_flop = !flip_flop;
	}
	destroySpatialAudio(session);
}
