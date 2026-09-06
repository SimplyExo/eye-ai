mod sys;
#[cfg(feature = "kalman-motion-benchmark")]
use sys::byte_track_BYTETracker_update_for_benchmark;
use sys::{
	byte_track_BYTETracker_create, byte_track_BYTETracker_destroy,
	byte_track_BYTETracker_set_max_time_lost, byte_track_BYTETracker_update, byte_track_Object,
	byte_track_Rect_float, byte_track_STrack, byte_track_STrack_array_destroy,
};

use std::{
	os::raw::{c_int, c_void},
	time::Duration,
};

pub type Rect = byte_track_Rect_float;
pub type Object = byte_track_Object;
pub type STrack = byte_track_STrack;

pub struct BYTETracker {
	tracker: *mut c_void,
}
unsafe impl Send for BYTETracker {}
unsafe impl Sync for BYTETracker {}
impl BYTETracker {
	pub const DEFAULT_MAX_TRACKING_TIME_SECONDS: f64 = 10.0;
	pub const DEFAULT_TRACK_THRESHOLD: f32 = 0.5;
	pub const DEFAULT_HIGH_THRESHOLD: f32 = 0.6;
	pub const DEFAULT_MATCH_THRESHOLD: f32 = 0.8;

	pub fn new(
		max_time_lost_seconds: f64,
		track_thresh: f32,
		high_thresh: f32,
		match_thresh: f32,
	) -> Self {
		Self {
			tracker: unsafe {
				byte_track_BYTETracker_create(
					max_time_lost_seconds,
					track_thresh,
					high_thresh,
					match_thresh,
				)
			},
		}
	}

	// TODO: do not have the c api alloc a array and then clone that again, make custom struct that destroys c api allocated array on drop
	/// Advances ByteTrack by the real monotonic duration since the previous
	/// detector/tracker update. Skipped source frames must not call this method.
	/// The FFI uses integer nanoseconds and saturates durations beyond `u64`.
	pub fn update(
		&mut self,
		objects: &[byte_track_Object],
		elapsed: Duration,
	) -> Vec<byte_track_STrack> {
		let elapsed_nanoseconds = elapsed.as_nanos().min(u64::MAX as u128) as u64;
		let mut out_stracks_ptr: *mut byte_track_STrack = std::ptr::null_mut();
		let mut out_stracks_len: c_int = 0;
		unsafe {
			byte_track_BYTETracker_update(
				self.tracker,
				objects.as_ptr(),
				objects.len() as c_int,
				elapsed_nanoseconds,
				&mut out_stracks_ptr,
				&mut out_stracks_len,
			);
			let stracks_vec =
				std::slice::from_raw_parts(out_stracks_ptr, out_stracks_len as usize).to_vec();
			byte_track_STrack_array_destroy(out_stracks_ptr);
			stracks_vec
		}
	}

	/// Runs the real ByteTrack association/update path with an isolated
	/// prediction strategy for the Kalman motion simulation.
	///
	/// `enable_motion_prediction = false` retains the last measured bounding box
	/// for association. `process_noise_scale` only scales the Kalman Q matrix
	/// and is ignored when prediction is disabled. Normal product code must use
	/// [`Self::update`].
	#[cfg(feature = "kalman-motion-benchmark")]
	pub fn update_for_benchmark(
		&mut self,
		objects: &[byte_track_Object],
		elapsed: Duration,
		enable_motion_prediction: bool,
		process_noise_scale: f32,
	) -> Vec<byte_track_STrack> {
		let elapsed_nanoseconds = elapsed.as_nanos().min(u64::MAX as u128) as u64;
		let mut out_stracks_ptr: *mut byte_track_STrack = std::ptr::null_mut();
		let mut out_stracks_len: c_int = 0;
		unsafe {
			byte_track_BYTETracker_update_for_benchmark(
				self.tracker,
				objects.as_ptr(),
				objects.len() as c_int,
				elapsed_nanoseconds,
				enable_motion_prediction,
				process_noise_scale,
				&mut out_stracks_ptr,
				&mut out_stracks_len,
			);
			let stracks_vec =
				std::slice::from_raw_parts(out_stracks_ptr, out_stracks_len as usize).to_vec();
			byte_track_STrack_array_destroy(out_stracks_ptr);
			stracks_vec
		}
	}

	pub fn set_max_time_lost_seconds(&mut self, max_time_lost_seconds: f64) {
		unsafe {
			byte_track_BYTETracker_set_max_time_lost(self.tracker, max_time_lost_seconds);
		}
	}
}
impl Drop for BYTETracker {
	fn drop(&mut self) {
		unsafe { byte_track_BYTETracker_destroy(self.tracker) };
	}
}
impl Default for BYTETracker {
	fn default() -> Self {
		Self::new(
			Self::DEFAULT_MAX_TRACKING_TIME_SECONDS,
			Self::DEFAULT_TRACK_THRESHOLD,
			Self::DEFAULT_HIGH_THRESHOLD,
			Self::DEFAULT_MATCH_THRESHOLD,
		)
	}
}

#[cfg(test)]
mod rect_iou_tests {
	use super::{Rect, sys::byte_track_Rect_float_calc_iou_for_testing};

	const EPSILON: f32 = 1e-5;

	fn rect(x: f32, y: f32, width: f32, height: f32) -> Rect {
		Rect::new(x, y, width, height)
	}

	fn native_iou(first: Rect, second: Rect) -> f32 {
		unsafe { byte_track_Rect_float_calc_iou_for_testing(first, second) }
	}

	fn assert_iou(actual: f32, expected: f32) {
		assert!(actual.is_finite(), "IoU must be finite, got {actual}");
		assert!(
			(actual - expected).abs() <= EPSILON,
			"expected {expected}, got {actual}"
		);
		assert!((0.0..=1.0).contains(&actual), "IoU must be in [0, 1]");
	}

	#[test]
	fn identical_boxes_have_iou_one() {
		let box_ = rect(0.10, 0.20, 0.30, 0.40);
		assert_iou(native_iou(box_, box_), 1.0);
	}

	#[test]
	fn separated_normalized_boxes_have_iou_zero() {
		assert_iou(
			native_iou(rect(0.00, 0.00, 0.10, 0.10), rect(0.80, 0.80, 0.10, 0.10)),
			0.0,
		);
	}

	#[test]
	fn partial_overlap_uses_continuous_area() {
		// Intersection = 0.10 * 0.20, union = 0.20 * 0.20 * 2 - intersection.
		assert_iou(
			native_iou(rect(0.10, 0.10, 0.20, 0.20), rect(0.20, 0.10, 0.20, 0.20)),
			1.0 / 3.0,
		);
	}

	#[test]
	fn contained_box_uses_the_larger_box_as_union() {
		// Intersection = 0.20 * 0.20, union = 0.60 * 0.60.
		assert_iou(
			native_iou(rect(0.10, 0.10, 0.60, 0.60), rect(0.20, 0.20, 0.20, 0.20)),
			1.0 / 9.0,
		);
	}

	#[test]
	fn tiny_and_typical_yolo_boxes_have_analytical_iou() {
		// Tiny normalized boxes: intersection = 0.005 * 0.01, union = 0.00015.
		assert_iou(
			native_iou(rect(0.10, 0.10, 0.01, 0.01), rect(0.105, 0.10, 0.01, 0.01)),
			1.0 / 3.0,
		);
		// Typical YOLO boxes: intersection = 0.15 * 0.40, union = 0.19.
		assert_iou(
			native_iou(rect(0.10, 0.10, 0.25, 0.50), rect(0.20, 0.20, 0.25, 0.50)),
			6.0 / 19.0,
		);
	}

	#[test]
	fn image_edges_and_degenerate_boxes_are_safe() {
		assert_iou(
			native_iou(rect(0.00, 0.00, 0.20, 0.20), rect(0.80, 0.80, 0.20, 0.20)),
			0.0,
		);
		assert_iou(
			native_iou(rect(0.80, 0.80, 0.20, 0.20), rect(0.80, 0.80, 0.20, 0.20)),
			1.0,
		);

		for invalid in [
			rect(0.10, 0.10, 0.00, 0.20),
			rect(0.10, 0.10, 0.20, 0.00),
			rect(0.10, 0.10, -0.20, 0.20),
			rect(f32::NAN, 0.10, 0.20, 0.20),
			rect(0.10, f32::INFINITY, 0.20, 0.20),
		] {
			assert_iou(native_iou(invalid, rect(0.10, 0.10, 0.20, 0.20)), 0.0);
		}
	}
}
