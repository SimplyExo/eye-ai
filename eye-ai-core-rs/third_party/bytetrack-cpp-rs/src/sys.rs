#![allow(non_camel_case_types)]

use std::os::raw::{c_double, c_float, c_int, c_void};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct byte_track_Rect_float {
	pub x: c_float,
	pub y: c_float,
	pub width: c_float,
	pub height: c_float,
}
impl byte_track_Rect_float {
	pub fn new(x: c_float, y: c_float, width: c_float, height: c_float) -> Self {
		Self {
			x,
			y,
			width,
			height,
		}
	}
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct byte_track_Object {
	pub rect: byte_track_Rect_float,
	pub label: c_int,
	pub prob: c_float,
}
impl byte_track_Object {
	pub fn new(rect: byte_track_Rect_float, label: c_int, prob: c_float) -> Self {
		Self { rect, label, prob }
	}
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum byte_track_STrackState {
	New = 0,
	Tracked = 1,
	Lost = 2,
	Removed = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct byte_track_STrack {
	pub rect: byte_track_Rect_float,
	pub state: byte_track_STrackState,
	pub is_activated: bool,
	pub score: c_float,
	pub track_id: c_int,
	pub frame_id: c_int,
	pub start_frame_id: c_int,
	pub tracklet_length: c_int,
	pub label: c_int,
}

unsafe extern "C" {
	pub unsafe fn byte_track_BYTETracker_create(
		max_time_lost_seconds: c_double,
		track_thresh: c_float,
		high_thresh: c_float,
		match_thresh: c_float,
	) -> *mut c_void;
	pub unsafe fn byte_track_BYTETracker_destroy(tracker: *mut c_void);
	pub unsafe fn byte_track_BYTETracker_update(
		tracker: *mut c_void,
		objects: *const byte_track_Object,
		num_objects: i32,
		elapsed_nanoseconds: u64,
		out_stracks: *mut *mut byte_track_STrack,
		out_num_stracks: *mut i32,
	);
	pub unsafe fn byte_track_STrack_array_destroy(stracks_array: *mut byte_track_STrack);
	pub unsafe fn byte_track_BYTETracker_set_max_time_lost(
		tracker: *mut c_void,
		max_time_lost_seconds: c_double,
	);
}
