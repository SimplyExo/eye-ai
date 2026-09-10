use super::*;
use std::time::{Duration, Instant};

pub(super) const HIGH_HZ: f32 = 15.0;
pub(super) const NORMAL_HZ: f32 = 10.0;
pub(super) const MEDIUM_HZ: f32 = 5.0;
pub(super) const LOW_HZ: f32 = 3.0;
pub(super) const CONFIDENCE: f32 = 0.9;
const BOX_WIDTH: f32 = 0.2;
const BOX_HEIGHT: f32 = 0.2;

#[derive(Debug, Clone, Copy)]
pub(super) struct TruthBox {
	pub center_x: f32,
	pub center_y: f32,
	pub width: f32,
	pub height: f32,
}

pub(super) fn detection_at(truth: TruthBox) -> DetectedObject {
	DetectedObject::new(
		"object".to_string(),
		0,
		CONFIDENCE,
		BoundingBox::new(truth.center_x, truth.center_y, truth.width, truth.height),
	)
}

pub(super) fn detection(center_x: f32) -> DetectedObject {
	detection_with_confidence(center_x, CONFIDENCE)
}

pub(super) fn detection_with_confidence(center_x: f32, confidence: f32) -> DetectedObject {
	let mut object = detection_at(TruthBox {
		center_x,
		center_y: 0.5,
		width: BOX_WIDTH,
		height: BOX_HEIGHT,
	});
	object.confidence = confidence;
	object
}

pub(super) fn update_after(
	tracker: &mut ObjectTracker<'_>,
	interval_seconds: f32,
	detections: Vec<DetectedObject>,
) -> Vec<TrackedObject> {
	update_after_duration(
		tracker,
		Duration::from_secs_f32(interval_seconds),
		detections,
	)
}

pub(super) fn update_after_duration(
	tracker: &mut ObjectTracker<'_>,
	interval: Duration,
	detections: Vec<DetectedObject>,
) -> Vec<TrackedObject> {
	let last_update = tracker.last_update.unwrap_or_else(|| {
		let initial_time = Instant::now();
		tracker.last_update = Some(initial_time);
		initial_time
	});
	tracker.update_at(detections, last_update + interval)
}

pub(super) fn validation_state(
	tracker: &ObjectTracker<'_>,
	tracking_id: i32,
) -> TrackValidationState {
	tracker
		.track_validations
		.get(&tracking_id)
		.expect("validation state should exist")
		.state
}

pub(super) fn tentative_visible_seconds(tracker: &ObjectTracker<'_>, tracking_id: i32) -> f32 {
	let TrackValidationState::Tentative {
		confidence_visible_seconds,
	} = validation_state(tracker, tracking_id)
	else {
		panic!("track should still be tentative");
	};
	confidence_visible_seconds
}

pub(super) fn repeated_interval(hz: f32, count: usize) -> Vec<f32> {
	vec![1.0 / hz; count]
}

pub(super) fn confirmation_time_at(hz: f32, confidence: f32) -> f32 {
	let profiling_frame = ProfilingFrame::new("confirmation_time");
	let mut tracker = ObjectTracker::new(vec!["object".to_string()], &profiling_frame);
	let interval_seconds = 1.0 / hz;
	let mut elapsed_seconds = 0.0;

	for _ in 0..60 {
		elapsed_seconds += interval_seconds;
		let output = update_after(
			&mut tracker,
			interval_seconds,
			vec![detection_with_confidence(0.5, confidence)],
		);
		if !output.is_empty() {
			return elapsed_seconds;
		}
	}
	panic!("continuous detection should eventually be confirmed");
}

pub(super) fn warm_visible_track(tracker: &mut ObjectTracker<'_>, hz: f32) -> i32 {
	let mut id = None;
	for _ in 0..12 {
		let output = update_after(tracker, 1.0 / hz, vec![detection(0.5)]);
		if let Some(object) = output.first() {
			id = Some(object.tracking_id);
		}
	}
	id.expect("track should be visible after warm-up")
}

pub(super) fn visible_id_after_detections(
	tracker: &mut ObjectTracker<'_>,
	hz: f32,
	count: usize,
) -> i32 {
	let mut id = None;
	for _ in 0..count {
		let output = update_after(tracker, 1.0 / hz, vec![detection(0.5)]);
		if let Some(object) = output.first() {
			id = Some(object.tracking_id);
		}
	}
	id.expect("track should become visible again")
}

pub(super) fn id_after_constant_rate_loss(hz: f32, lost_seconds: f32) -> (i32, i32) {
	let profiling_frame = ProfilingFrame::new("constant_rate_loss");
	let mut tracker = ObjectTracker::new(vec!["object".to_string()], &profiling_frame);
	let original_id = warm_visible_track(&mut tracker, hz);
	let intervals = segmented_loss_intervals(Duration::from_secs_f32(lost_seconds), &[], hz);
	for &interval in &intervals[..intervals.len() - 1] {
		update_after_duration(&mut tracker, interval, Vec::new());
	}
	let immediate_output = update_after_duration(
		&mut tracker,
		*intervals.last().unwrap(),
		vec![detection(0.5)],
	);
	let reacquired_id = immediate_output
		.first()
		.map(|object| object.tracking_id)
		.unwrap_or_else(|| visible_id_after_detections(&mut tracker, hz, 12));
	(original_id, reacquired_id)
}

fn append_fixed_rate_intervals(intervals: &mut Vec<Duration>, duration: Duration, hz: f32) {
	let step = Duration::from_secs_f64(1.0 / f64::from(hz));
	let mut remaining = duration;
	while remaining > step {
		intervals.push(step);
		remaining -= step;
	}
	if !remaining.is_zero() {
		intervals.push(remaining);
	}
}

pub(super) fn segmented_loss_intervals(
	total: Duration,
	segments: &[(Duration, f32)],
	final_hz: f32,
) -> Vec<Duration> {
	let mut intervals = Vec::new();
	let mut remaining = total;
	for &(requested_duration, hz) in segments {
		let segment_duration = requested_duration.min(remaining);
		append_fixed_rate_intervals(&mut intervals, segment_duration, hz);
		remaining -= segment_duration;
		if remaining.is_zero() {
			break;
		}
	}
	append_fixed_rate_intervals(&mut intervals, remaining, final_hz);
	intervals
}

pub(super) fn irregular_loss_intervals(total: Duration) -> Vec<Duration> {
	let pattern = [
		Duration::from_secs_f64(1.0 / 15.0),
		Duration::from_secs_f64(1.0 / 10.0),
		Duration::from_secs_f64(1.0 / 4.0),
		Duration::from_secs_f64(1.0 / 3.0),
		Duration::from_secs_f64(1.0 / 8.0),
	];
	let mut intervals = Vec::new();
	let mut remaining = total;
	for interval in pattern.into_iter().cycle() {
		if remaining.is_zero() {
			break;
		}
		let interval = interval.min(remaining);
		intervals.push(interval);
		remaining -= interval;
	}
	intervals
}

pub(super) fn id_after_loss_intervals(intervals: &[Duration]) -> (i32, i32, bool) {
	assert!(!intervals.is_empty());
	let profiling_frame = ProfilingFrame::new("real_time_loss_schedule");
	let mut tracker = ObjectTracker::new(vec!["object".to_string()], &profiling_frame);
	let original_id = warm_visible_track(&mut tracker, HIGH_HZ);

	for &interval in &intervals[..intervals.len() - 1] {
		update_after_duration(&mut tracker, interval, Vec::new());
	}
	let immediate_output = update_after_duration(
		&mut tracker,
		*intervals.last().unwrap(),
		vec![detection(0.5)],
	);
	let immediate_reassociation = immediate_output.first().map(|object| object.tracking_id);
	let reacquired_id = immediate_reassociation
		.unwrap_or_else(|| visible_id_after_detections(&mut tracker, HIGH_HZ, 12));
	if reacquired_id == original_id {
		assert_eq!(
			validation_state(&tracker, original_id),
			TrackValidationState::Confirmed
		);
	} else {
		assert!(!tracker.track_validations.contains_key(&original_id));
		assert_eq!(tracker.track_validations.len(), 1);
	}
	(
		original_id,
		reacquired_id,
		immediate_reassociation == Some(original_id),
	)
}
