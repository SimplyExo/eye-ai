use super::*;
use std::collections::BTreeSet;
use std::time::{Duration, Instant};

const HIGH_HZ: f32 = 15.0;
const NORMAL_HZ: f32 = 10.0;
const MEDIUM_HZ: f32 = 5.0;
const LOW_HZ: f32 = 3.0;
const CONFIDENCE: f32 = 0.9;
const BOX_WIDTH: f32 = 0.2;
const BOX_HEIGHT: f32 = 0.2;
const MULTI_BOX_WIDTH: f32 = 0.16;
const MULTI_BOX_HEIGHT: f32 = 0.16;
const MAX_ASSIGNMENT_ERROR: f32 = 0.30;

#[derive(Debug, Clone, Copy)]
struct TruthBox {
	center_x: f32,
	center_y: f32,
	width: f32,
	height: f32,
}

#[derive(Debug)]
struct ScenarioMetrics {
	name: &'static str,
	updates: usize,
	detection_updates: usize,
	visible_updates: usize,
	post_visible_misses_with_detection: usize,
	max_output_count: usize,
	unique_ids: BTreeSet<i32>,
	id_switches: usize,
	first_visible_seconds: Option<f32>,
	max_center_error: f32,
}

#[derive(Debug)]
struct MultiScenarioMetrics {
	name: &'static str,
	updates: usize,
	visible_truth_updates: usize,
	visible_by_truth: Vec<usize>,
	post_visible_misses_by_truth: Vec<usize>,
	reassociations_same_id: usize,
	max_output_count: usize,
	duplicate_id_updates: usize,
	unique_ids: BTreeSet<i32>,
	id_switches: usize,
	id_switches_by_truth: Vec<usize>,
	first_ids: Vec<Option<i32>>,
	last_ids: Vec<Option<i32>>,
	first_visible_seconds: Vec<Option<f32>>,
	max_center_error: f32,
}

fn detection_at(truth: TruthBox) -> DetectedObject {
	DetectedObject::new(
		"object".to_string(),
		0,
		CONFIDENCE,
		BoundingBox::new(truth.center_x, truth.center_y, truth.width, truth.height),
	)
}

fn detection(center_x: f32) -> DetectedObject {
	detection_with_confidence(center_x, CONFIDENCE)
}

fn detection_with_confidence(center_x: f32, confidence: f32) -> DetectedObject {
	let mut object = detection_at(TruthBox {
		center_x,
		center_y: 0.5,
		width: BOX_WIDTH,
		height: BOX_HEIGHT,
	});
	object.confidence = confidence;
	object
}

fn update_after(
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

fn update_after_duration(
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

fn validation_state(tracker: &ObjectTracker<'_>, tracking_id: i32) -> TrackValidationState {
	tracker
		.track_validations
		.get(&tracking_id)
		.expect("validation state should exist")
		.state
}

fn tentative_visible_seconds(tracker: &ObjectTracker<'_>, tracking_id: i32) -> f32 {
	let TrackValidationState::Tentative {
		confidence_visible_seconds,
	} = validation_state(tracker, tracking_id)
	else {
		panic!("track should still be tentative");
	};
	confidence_visible_seconds
}

fn run_scenario(
	name: &'static str,
	intervals_seconds: &[f32],
	detection_present: &[bool],
	velocity_per_second: f32,
) -> ScenarioMetrics {
	assert_eq!(intervals_seconds.len(), detection_present.len());
	let profiling_frame = ProfilingFrame::new(name);
	let mut tracker = ObjectTracker::new(vec!["object".to_string()], &profiling_frame);
	let mut elapsed_seconds = 0.0;
	let mut detection_updates = 0;
	let mut visible_updates = 0;
	let mut post_visible_misses_with_detection = 0;
	let mut max_output_count = 0;
	let mut has_been_visible = false;
	let mut first_visible_seconds = None;
	let mut unique_ids = BTreeSet::new();
	let mut previous_visible_id = None;
	let mut id_switches = 0;
	let mut max_center_error = 0.0_f32;

	for (&interval_seconds, &present) in intervals_seconds.iter().zip(detection_present.iter()) {
		elapsed_seconds += interval_seconds;
		let expected_center_x = 0.2 + velocity_per_second * elapsed_seconds;
		let detections = if present {
			detection_updates += 1;
			vec![detection(expected_center_x)]
		} else {
			Vec::new()
		};
		let output = update_after(&mut tracker, interval_seconds, detections);
		max_output_count = max_output_count.max(output.len());

		if present && output.is_empty() && has_been_visible {
			post_visible_misses_with_detection += 1;
		}
		for object in &output {
			unique_ids.insert(object.tracking_id);
		}
		if let Some(object) = output.iter().min_by(|a, b| {
			let a_error = (a.object.bbox.center_x - expected_center_x).abs();
			let b_error = (b.object.bbox.center_x - expected_center_x).abs();
			a_error.total_cmp(&b_error)
		}) {
			visible_updates += 1;
			if first_visible_seconds.is_none() {
				first_visible_seconds = Some(elapsed_seconds);
			}
			has_been_visible = true;
			if previous_visible_id.is_some_and(|id| id != object.tracking_id) {
				id_switches += 1;
			}
			previous_visible_id = Some(object.tracking_id);
			max_center_error =
				max_center_error.max((object.object.bbox.center_x - expected_center_x).abs());
		}
	}

	ScenarioMetrics {
		name,
		updates: intervals_seconds.len(),
		detection_updates,
		visible_updates,
		post_visible_misses_with_detection,
		max_output_count,
		unique_ids,
		id_switches,
		first_visible_seconds,
		max_center_error,
	}
}

fn repeated_interval(hz: f32, count: usize) -> Vec<f32> {
	vec![1.0 / hz; count]
}

fn center_error(output: &TrackedObject, truth: TruthBox) -> f32 {
	(output.object.bbox.center_x - truth.center_x)
		.hypot(output.object.bbox.center_y - truth.center_y)
}

/// Associates outputs with the two synthetic truth boxes without using any
/// tracker internals. The two-object permutation makes the crossing case
/// reproducible while the distance limit prevents a stale/ghost output from
/// being counted as a valid re-association.
fn assign_multi_outputs(output: &[TrackedObject], truths: &[TruthBox]) -> Vec<Option<usize>> {
	let mut assignments = vec![None; truths.len()];
	if truths.len() == 2 && output.len() >= 2 {
		let mut best = None;
		for first_output in 0..output.len() {
			for second_output in 0..output.len() {
				if first_output == second_output {
					continue;
				}
				let first_error = center_error(&output[first_output], truths[0]);
				let second_error = center_error(&output[second_output], truths[1]);
				let total_error = first_error + second_error;
				if best.is_none_or(|(best_error, _, _, _, _)| total_error < best_error) {
					best = Some((
						total_error,
						first_output,
						second_output,
						first_error,
						second_error,
					));
				}
			}
		}
		if let Some((_, first_output, second_output, first_error, second_error)) = best {
			if first_error <= MAX_ASSIGNMENT_ERROR {
				assignments[0] = Some(first_output);
			}
			if second_error <= MAX_ASSIGNMENT_ERROR {
				assignments[1] = Some(second_output);
			}
		}
		return assignments;
	}

	let mut used_outputs = vec![false; output.len()];
	for (truth_index, truth) in truths.iter().copied().enumerate() {
		let Some((output_index, error)) = output
			.iter()
			.enumerate()
			.filter(|(index, _)| !used_outputs[*index])
			.map(|(index, object)| (index, center_error(object, truth)))
			.min_by(|(_, a), (_, b)| a.total_cmp(b))
		else {
			continue;
		};
		if error <= MAX_ASSIGNMENT_ERROR {
			used_outputs[output_index] = true;
			assignments[truth_index] = Some(output_index);
		}
	}
	assignments
}

fn run_multi_scenario<F>(
	name: &'static str,
	intervals_seconds: &[f32],
	mut truths_at: F,
) -> MultiScenarioMetrics
where
	F: FnMut(f32) -> Vec<TruthBox>,
{
	let profiling_frame = ProfilingFrame::new(name);
	let mut tracker = ObjectTracker::new(vec!["object".to_string()], &profiling_frame);
	let mut elapsed_seconds = 0.0;
	let mut visible_truth_updates = 0;
	let mut max_output_count = 0;
	let mut duplicate_id_updates = 0;
	let mut unique_ids = BTreeSet::new();
	let mut max_center_error = 0.0_f32;
	let mut visible_by_truth = Vec::new();
	let mut post_visible_misses_by_truth = Vec::new();
	let mut reassociations_same_id = 0;
	let mut id_switches = 0;
	let mut id_switches_by_truth = Vec::new();
	let mut first_ids = Vec::new();
	let mut last_ids = Vec::new();
	let mut first_visible_seconds = Vec::new();
	let mut had_gap = Vec::new();

	for &interval_seconds in intervals_seconds {
		elapsed_seconds += interval_seconds;
		let truths = truths_at(elapsed_seconds);
		if visible_by_truth.is_empty() {
			let truth_count = truths.len();
			visible_by_truth = vec![0; truth_count];
			post_visible_misses_by_truth = vec![0; truth_count];
			id_switches_by_truth = vec![0; truth_count];
			first_ids = vec![None; truth_count];
			last_ids = vec![None; truth_count];
			first_visible_seconds = vec![None; truth_count];
			had_gap = vec![false; truth_count];
		}
		assert_eq!(truths.len(), visible_by_truth.len());

		let detections = truths.iter().copied().map(detection_at).collect();
		let output = update_after(&mut tracker, interval_seconds, detections);
		max_output_count = max_output_count.max(output.len());
		let mut ids_in_update = BTreeSet::new();
		for object in &output {
			unique_ids.insert(object.tracking_id);
			if !ids_in_update.insert(object.tracking_id) {
				duplicate_id_updates += 1;
			}
		}

		let assignments = assign_multi_outputs(&output, &truths);
		for (truth_index, assignment) in assignments.into_iter().enumerate() {
			let Some(output_index) = assignment else {
				if first_ids[truth_index].is_some() {
					post_visible_misses_by_truth[truth_index] += 1;
					had_gap[truth_index] = true;
				}
				continue;
			};

			let tracked = &output[output_index];
			let tracking_id = tracked.tracking_id;
			let error = center_error(tracked, truths[truth_index]);
			max_center_error = max_center_error.max(error);
			visible_truth_updates += 1;
			visible_by_truth[truth_index] += 1;

			if first_ids[truth_index].is_none() {
				first_ids[truth_index] = Some(tracking_id);
				first_visible_seconds[truth_index] = Some(elapsed_seconds);
			} else {
				if last_ids[truth_index] != Some(tracking_id) {
					id_switches += 1;
					id_switches_by_truth[truth_index] += 1;
				}
				if had_gap[truth_index] && last_ids[truth_index] == Some(tracking_id) {
					reassociations_same_id += 1;
				}
			}
			last_ids[truth_index] = Some(tracking_id);
			had_gap[truth_index] = false;
		}
	}

	MultiScenarioMetrics {
		name,
		updates: intervals_seconds.len(),
		visible_truth_updates,
		visible_by_truth,
		post_visible_misses_by_truth,
		reassociations_same_id,
		max_output_count,
		duplicate_id_updates,
		unique_ids,
		id_switches,
		id_switches_by_truth,
		first_ids,
		last_ids,
		first_visible_seconds,
		max_center_error,
	}
}

fn print_multi_metrics(metrics: &MultiScenarioMetrics) {
	println!(
		"{}\tupdates={}\tvisible_truth_updates={}\tvisible_by_truth={:?}\tpost_visible_misses_by_truth={:?}\treassociations_same_id={}\tmax_output_count={}\tduplicate_id_updates={}\tunique_ids={:?}\tid_switches={}\tid_switches_by_truth={:?}\tfirst_ids={:?}\tlast_ids={:?}\tfirst_visible_s={:?}\tmax_center_error={:.6}",
		metrics.name,
		metrics.updates,
		metrics.visible_truth_updates,
		metrics.visible_by_truth,
		metrics.post_visible_misses_by_truth,
		metrics.reassociations_same_id,
		metrics.max_output_count,
		metrics.duplicate_id_updates,
		metrics.unique_ids,
		metrics.id_switches,
		metrics.id_switches_by_truth,
		metrics.first_ids,
		metrics.last_ids,
		metrics.first_visible_seconds,
		metrics.max_center_error,
	);
}

fn crossing_truths(elapsed_seconds: f32) -> Vec<TruthBox> {
	let first_x = 0.25 + 0.12 * elapsed_seconds;
	let second_x = 0.75 - 0.12 * elapsed_seconds;
	vec![
		TruthBox {
			center_x: first_x,
			center_y: 0.5,
			width: MULTI_BOX_WIDTH,
			height: MULTI_BOX_HEIGHT,
		},
		TruthBox {
			center_x: second_x,
			center_y: 0.5,
			width: MULTI_BOX_WIDTH,
			height: MULTI_BOX_HEIGHT,
		},
	]
}

fn close_then_separate_truths(elapsed_seconds: f32) -> Vec<TruthBox> {
	let separation_seconds = (elapsed_seconds - 1.0).clamp(0.0, 3.0);
	let half_gap = 0.10 + 0.07 * separation_seconds;
	vec![
		TruthBox {
			center_x: 0.5 - half_gap,
			center_y: 0.5,
			width: MULTI_BOX_WIDTH,
			height: MULTI_BOX_HEIGHT,
		},
		TruthBox {
			center_x: 0.5 + half_gap,
			center_y: 0.5,
			width: MULTI_BOX_WIDTH,
			height: MULTI_BOX_HEIGHT,
		},
	]
}

fn print_metrics(metrics: &ScenarioMetrics) {
	println!(
		"{}\tupdates={}\tdetections={}\tvisible={}\tpost_visible_misses={}\tmax_output_count={}\tunique_ids={:?}\tid_switches={}\tfirst_visible_s={:?}\tmax_center_error={:.6}",
		metrics.name,
		metrics.updates,
		metrics.detection_updates,
		metrics.visible_updates,
		metrics.post_visible_misses_with_detection,
		metrics.max_output_count,
		metrics.unique_ids,
		metrics.id_switches,
		metrics.first_visible_seconds,
		metrics.max_center_error,
	);
}

fn confirmation_time_at(hz: f32, confidence: f32) -> f32 {
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

fn warm_visible_track(tracker: &mut ObjectTracker<'_>, hz: f32) -> i32 {
	let mut id = None;
	for _ in 0..12 {
		let output = update_after(tracker, 1.0 / hz, vec![detection(0.5)]);
		if let Some(object) = output.first() {
			id = Some(object.tracking_id);
		}
	}
	id.expect("track should be visible after warm-up")
}

fn visible_id_after_detections(tracker: &mut ObjectTracker<'_>, hz: f32, count: usize) -> i32 {
	let mut id = None;
	for _ in 0..count {
		let output = update_after(tracker, 1.0 / hz, vec![detection(0.5)]);
		if let Some(object) = output.first() {
			id = Some(object.tracking_id);
		}
	}
	id.expect("track should become visible again")
}

fn id_after_constant_rate_loss(hz: f32, lost_seconds: f32) -> (i32, i32) {
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

fn segmented_loss_intervals(
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

fn irregular_loss_intervals(total: Duration) -> Vec<Duration> {
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

fn id_after_loss_intervals(intervals: &[Duration]) -> (i32, i32, bool) {
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

mod cadence;
mod lost_lifetime;
mod multi_object;
mod validation;
