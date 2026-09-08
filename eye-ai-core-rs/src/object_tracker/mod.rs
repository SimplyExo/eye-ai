use bytetrack_cpp_rs::BYTETracker;
use eye_ai_core_rs_profiling_attribute::profile_function;
use std::{
	collections::HashMap,
	time::{Duration, Instant},
};

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

#[derive(Debug, Clone, Copy, PartialEq)]
enum TrackValidationState {
	Tentative { confidence_visible_seconds: f32 },
	Confirmed,
}

#[derive(Debug, Clone, Copy)]
struct TrackValidation {
	state: TrackValidationState,
	last_seen: Instant,
	last_seen_update: u64,
}

pub struct ObjectTracker<'a> {
	labels: Vec<String>,
	tracker: BYTETracker,
	last_update: Option<Instant>,
	update_number: u64,
	track_validations: HashMap<i32, TrackValidation>,
	profiling_frame: &'a ProfilingFrame,
}
impl<'a> std::fmt::Debug for ObjectTracker<'a> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("ObjectTracker")
			.field("labels", &self.labels)
			.field("last_update", &self.last_update)
			.field("update_number", &self.update_number)
			.field("track_validations", &self.track_validations)
			.field("profiling_frame", &self.profiling_frame)
			.finish_non_exhaustive()
	}
}
impl<'a> ObjectTracker<'a> {
	/// For how many seconds a 100% confident tracked observation needs to be
	/// visible before it is considered valid.
	pub const MIN_WAITING_PREDICTION_TIME_BEFORE_VALID: f32 = 0.5;

	pub fn new(labels: Vec<String>, profiling_frame: &'a ProfilingFrame) -> Self {
		Self {
			labels,
			tracker: BYTETracker::default(),
			last_update: None,
			update_number: 0,
			track_validations: HashMap::new(),
			profiling_frame,
		}
	}

	fn is_track_confirmed(
		&mut self,
		tracking_id: i32,
		confidence: f32,
		now: Instant,
		update_duration: Duration,
	) -> bool {
		let validation = self
			.track_validations
			.entry(tracking_id)
			.or_insert(TrackValidation {
				state: TrackValidationState::Tentative {
					confidence_visible_seconds: 0.0,
				},
				last_seen: now,
				last_seen_update: self.update_number,
			});

		let was_seen_in_previous_update =
			validation.last_seen_update == self.update_number.wrapping_sub(1);
		validation.last_seen = now;
		validation.last_seen_update = self.update_number;

		let TrackValidationState::Tentative {
			confidence_visible_seconds,
		} = &mut validation.state
		else {
			return true;
		};

		// A duration is only evidence of visibility if this ID was observed in the
		// previous tracker update. The first ByteTrack output and a longer
		// unobserved gap are never credited.
		let max_unobserved_interval =
			Duration::from_secs_f32(Self::MIN_WAITING_PREDICTION_TIME_BEFORE_VALID);
		if was_seen_in_previous_update && update_duration <= max_unobserved_interval {
			let bounded_confidence = if confidence.is_finite() {
				confidence.clamp(0.0, 1.0)
			} else {
				0.0
			};
			*confidence_visible_seconds += bounded_confidence * update_duration.as_secs_f32();
		}

		if *confidence_visible_seconds >= Self::MIN_WAITING_PREDICTION_TIME_BEFORE_VALID {
			validation.state = TrackValidationState::Confirmed;
			true
		} else {
			false
		}
	}

	fn cleanup_stale_track_validations(&mut self, now: Instant) {
		let nominal_maximum_track_lifetime =
			Duration::from_secs_f64(BYTETracker::DEFAULT_MAX_TRACKING_TIME_SECONDS);
		self.track_validations.retain(|_, validation| {
			now.saturating_duration_since(validation.last_seen) <= nominal_maximum_track_lifetime
		});
	}

	#[profile_function("self.profiling_frame")]
	pub fn update(&mut self, detected_objects: Vec<DetectedObject>) -> Vec<TrackedObject> {
		self.update_at(detected_objects, Instant::now())
	}

	fn update_at(
		&mut self,
		detected_objects: Vec<DetectedObject>,
		now: Instant,
	) -> Vec<TrackedObject> {
		// `Instant` is monotonic. The first update has no predecessor and therefore
		// advances native tracking by zero; there cannot be an existing track to
		// predict or expire at that point.
		let update_duration = self.last_update.map_or(Duration::ZERO, |last_update| {
			now.saturating_duration_since(last_update)
		});
		self.last_update = Some(now);
		self.update_number = self.update_number.wrapping_add(1);

		let byte_track_objects = detected_objects
			.into_iter()
			.map(|detected_object| detected_object.into())
			.collect::<Vec<bytetrack_cpp_rs::Object>>();

		let byte_track_tracked_objects = self.tracker.update(&byte_track_objects, update_duration);

		let mut tracked_objects = Vec::with_capacity(byte_track_tracked_objects.len());
		for byte_track_tracked_object in byte_track_tracked_objects {
			let label = byte_track_tracked_object.label;
			if label < 0 {
				continue;
			}
			let Some(label) = self.labels.get(label as usize).cloned() else {
				continue;
			};
			let tracking_id = byte_track_tracked_object.track_id;

			if !self.is_track_confirmed(
				tracking_id,
				byte_track_tracked_object.score,
				now,
				update_duration,
			) {
				continue;
			}

			tracked_objects.push(TrackedObject {
				object: DetectedObject::new(
					label,
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
		self.cleanup_stale_track_validations(now);
		tracked_objects
	}
}

#[cfg(test)]
mod tests;
