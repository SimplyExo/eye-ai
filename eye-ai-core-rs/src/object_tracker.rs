use bytetrack_cpp_rs::BYTETracker;
use eye_ai_core_rs_profiling_attribute::profile_function;
use std::{collections::HashMap, time::Instant};

use crate::{BoundingBox, DetectedObject, ProfilingFrame};

#[derive(Debug, Clone)]
pub struct TrackedObject {
	pub object: DetectedObject,
	pub tracking_id: i32,
}
impl TrackedObject {
	pub fn new(object: DetectedObject, tracking_id: i32) -> Self {
		Self {
			object,
			tracking_id,
		}
	}
}

pub struct ObjectTracker<'a> {
	labels: Vec<String>,
	tracker: BYTETracker,
	last_update: Instant,
	/// sums up all the confidence of a tracked object
	tracked_object_valid_score: HashMap<i32, f32>,
	profiling_frame: &'a ProfilingFrame,
}
impl<'a> ObjectTracker<'a> {
	/// For how many seconds a 100% confident prediction needs to be tracked in
	/// order to be considered valid
	pub const MIN_WAITING_PREDICTION_TIME_BEFORE_VALID: f32 = 0.5;

	pub fn new(labels: Vec<String>, profiling_frame: &'a ProfilingFrame) -> Self {
		Self {
			labels,
			tracker: BYTETracker::default(),
			last_update: Instant::now(),
			tracked_object_valid_score: HashMap::new(),
			profiling_frame,
		}
	}

	#[profile_function("self.profiling_frame")]
	pub fn update(&mut self, detected_objects: Vec<DetectedObject>) -> Vec<TrackedObject> {
		let now = Instant::now();
		let update_duration = now - self.last_update;
		self.last_update = now;

		let frame_rate = 1.0 / update_duration.as_secs_f32();
		self.tracker
			.set_max_time_lost_seconds(BYTETracker::DEFAULT_MAX_TRACKING_TIME_SECONDS, frame_rate);

		let byte_track_objects = detected_objects
			.into_iter()
			.map(|detected_object| detected_object.into())
			.collect::<Vec<bytetrack_cpp_rs::Object>>();

		let byte_track_tracked_objects = self.tracker.update(&byte_track_objects);

		let mut tracked_objects = Vec::with_capacity(byte_track_tracked_objects.len());
		let min_valid_prediction_score: f32 =
			Self::MIN_WAITING_PREDICTION_TIME_BEFORE_VALID * frame_rate;
		for byte_track_tracked_object in byte_track_tracked_objects {
			let label = byte_track_tracked_object.label;
			if label < 0 {
				continue;
			}
			let Some(label) = self.labels.get(label as usize) else {
				continue;
			};
			let tracking_id = byte_track_tracked_object.track_id;

			let valid_score: &mut f32 = self
				.tracked_object_valid_score
				.entry(tracking_id)
				.or_insert(0.0);
			*valid_score += byte_track_tracked_object.score;
			if *valid_score < min_valid_prediction_score {
				continue;
			}

			tracked_objects.push(TrackedObject {
				object: DetectedObject::new(
					label.clone(),
					byte_track_tracked_object.label as usize,
					byte_track_tracked_object.score,
					BoundingBox::from_x_y_w_h(
						byte_track_tracked_object.rect.x,
						byte_track_tracked_object.rect.y,
						byte_track_tracked_object.rect.width,
						byte_track_tracked_object.rect.height,
					),
				),
				tracking_id,
			});
		}
		tracked_objects
	}
}
