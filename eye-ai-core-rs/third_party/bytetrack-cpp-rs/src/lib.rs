mod sys;
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
