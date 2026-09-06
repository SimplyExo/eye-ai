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
	pub const MIN_WAITING_PREDICTION_TIME_BEFORE_VALID: f32 = 0.45;

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

	/// Starts a fresh tracking epoch without reloading the detector/model.
	///
	/// Replacing `BYTETracker` clears its Kalman/lost/ID state. The Rust state
	/// is reset alongside it so no validation or timing evidence can cross the
	/// epoch boundary. Labels and the profiling frame are intentionally kept.
	pub fn reset(&mut self) {
		self.tracker = BYTETracker::default();
		self.last_update = None;
		self.update_number = 0;
		self.track_validations.clear();
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
mod cadence_tests {
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

		for (&interval_seconds, &present) in intervals_seconds.iter().zip(detection_present.iter())
		{
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

	#[test]
	fn cadence_characterization_matrix() {
		let a_intervals = repeated_interval(HIGH_HZ, 30);
		let a_present = vec![true; a_intervals.len()];

		let b_intervals = repeated_interval(LOW_HZ, 36);
		let b_present = vec![true; b_intervals.len()];

		let mut c_intervals = repeated_interval(HIGH_HZ, 12);
		c_intervals.extend(repeated_interval(MEDIUM_HZ, 6));
		c_intervals.extend(repeated_interval(HIGH_HZ, 15));
		let c_present = vec![true; c_intervals.len()];

		let mut c3_intervals = repeated_interval(HIGH_HZ, 12);
		c3_intervals.extend(repeated_interval(LOW_HZ, 6));
		c3_intervals.extend(repeated_interval(HIGH_HZ, 15));
		let c3_present = vec![true; c3_intervals.len()];

		let d_intervals = repeated_interval(HIGH_HZ, 30);
		let d_present = vec![true; d_intervals.len()];

		let mut e_intervals = repeated_interval(HIGH_HZ, 12);
		e_intervals.extend(repeated_interval(LOW_HZ, 6));
		e_intervals.extend(repeated_interval(HIGH_HZ, 15));
		let e_present = vec![true; e_intervals.len()];

		let f_intervals = repeated_interval(HIGH_HZ, 23);
		let mut f_present = vec![true; f_intervals.len()];
		f_present[12] = false;

		let g_intervals = repeated_interval(LOW_HZ, 10);
		let mut g_present = vec![true; g_intervals.len()];
		g_present[4] = false;

		let mut h_intervals = repeated_interval(HIGH_HZ, 12);
		h_intervals.push(12.0);
		h_intervals.extend(repeated_interval(HIGH_HZ, 12));
		let h_present = vec![true; h_intervals.len()];

		let mut c2_intervals = repeated_interval(NORMAL_HZ, 12);
		c2_intervals.extend(repeated_interval(LOW_HZ, 6));
		c2_intervals.extend(repeated_interval(NORMAL_HZ, 12));
		let c2_present = vec![true; c2_intervals.len()];

		let mut irregular_intervals = repeated_interval(HIGH_HZ, 12);
		irregular_intervals
			.extend([1.0 / 10.0, 1.0 / 4.0, 1.0 / 15.0, 1.0 / 3.0, 1.0 / 8.0].repeat(3));
		let irregular_present = vec![true; irregular_intervals.len()];

		let mut low_to_high_intervals = repeated_interval(LOW_HZ, 6);
		low_to_high_intervals.extend(repeated_interval(HIGH_HZ, 15));
		let low_to_high_present = vec![true; low_to_high_intervals.len()];

		let scenarios = [
			run_scenario("A_constant_high", &a_intervals, &a_present, 0.0),
			run_scenario("B_constant_low", &b_intervals, &b_present, 0.0),
			run_scenario("C_constant_high_low_high", &c_intervals, &c_present, 0.0),
			run_scenario(
				"C3_constant_high_3_low_high",
				&c3_intervals,
				&c3_present,
				0.0,
			),
			run_scenario("D_linear_constant_high", &d_intervals, &d_present, 0.6),
			run_scenario("E_linear_high_low_high", &e_intervals, &e_present, 0.6),
			run_scenario("F_one_miss_high", &f_intervals, &f_present, 0.0),
			run_scenario("G_one_miss_low", &g_intervals, &g_present, 0.0),
			run_scenario("H_twelve_second_pause", &h_intervals, &h_present, 0.0),
			run_scenario("C2_constant_10_3_10", &c2_intervals, &c2_present, 0.0),
			run_scenario(
				"I_linear_irregular_latency",
				&irregular_intervals,
				&irregular_present,
				0.6,
			),
			run_scenario(
				"J_stationary_3_to_15",
				&low_to_high_intervals,
				&low_to_high_present,
				0.0,
			),
			run_scenario("K_fast_constant_high", &d_intervals, &d_present, 1.2),
			run_scenario("L_fast_high_low_high", &e_intervals, &e_present, 1.2),
			run_scenario("M_linear_constant_low", &b_intervals, &b_present, 0.6),
			run_scenario(
				"N_linear_3_to_15",
				&low_to_high_intervals,
				&low_to_high_present,
				0.6,
			),
		];

		for metrics in &scenarios {
			print_metrics(metrics);
		}

		// Stationary tracks keep their ID across immediate cadence changes.
		for metrics in [
			&scenarios[0],  // 15 Hz stationary
			&scenarios[1],  // 3 Hz stationary
			&scenarios[2],  // 15 -> 5 -> 15 Hz stationary
			&scenarios[3],  // 15 -> 3 -> 15 Hz stationary
			&scenarios[9],  // 10 -> 3 -> 10 Hz stationary
			&scenarios[11], // 3 -> 15 Hz stationary
		] {
			assert_eq!(metrics.unique_ids.len(), 1, "{}", metrics.name);
			assert_eq!(metrics.id_switches, 0, "{}", metrics.name);
		}

		let long_pause = &scenarios[8];
		assert_eq!(long_pause.unique_ids.len(), 2, "{}", long_pause.name);
		assert_eq!(long_pause.id_switches, 1, "{}", long_pause.name);

		for metrics in [&scenarios[4], &scenarios[5], &scenarios[10]] {
			assert_eq!(metrics.unique_ids.len(), 1, "{}", metrics.name);
			assert_eq!(metrics.id_switches, 0, "{}", metrics.name);
			assert!(
				metrics.max_center_error < 0.01,
				"variable-dt regression in {}: {}",
				metrics.name,
				metrics.max_center_error
			);
		}
		for metrics in [&scenarios[12], &scenarios[13]] {
			assert_eq!(metrics.unique_ids.len(), 1, "{}", metrics.name);
			assert_eq!(metrics.id_switches, 0, "{}", metrics.name);
			assert!(metrics.max_center_error < 0.02, "{}", metrics.name);
		}
		for metrics in [&scenarios[14], &scenarios[15]] {
			assert_eq!(metrics.unique_ids.len(), 1, "{}", metrics.name);
			assert_eq!(metrics.id_switches, 0, "{}", metrics.name);
			assert!(metrics.max_center_error < 0.02, "{}", metrics.name);
		}
	}

	#[test]
	fn confirmed_track_stays_visible_after_rate_increase() {
		let profiling_frame = ProfilingFrame::new("validation_gate_rate_change");
		let mut tracker = ObjectTracker::new(vec!["object".to_string()], &profiling_frame);
		let mut visible = Vec::new();
		let mut ids = Vec::new();

		for interval in repeated_interval(LOW_HZ, 3)
			.into_iter()
			.chain(repeated_interval(HIGH_HZ, 7))
		{
			let output = update_after(&mut tracker, interval, vec![detection(0.5)]);
			visible.push(!output.is_empty());
			if let Some(object) = output.first() {
				ids.push(object.tracking_id);
			}
		}

		println!("validation_latch_low_to_high\tvisible={visible:?}\tids={ids:?}");
		assert!(visible[2], "track should pass the low-rate validation gate");
		assert!(
			visible[3..].iter().all(|is_visible| *is_visible),
			"a confirmed track must remain visible after the FPS increase"
		);
		assert!(ids.iter().all(|id| *id == ids[0]), "ByteTrack kept the ID");
		assert_eq!(
			validation_state(&tracker, ids[0]),
			TrackValidationState::Confirmed
		);
	}

	#[test]
	fn short_single_detection_stays_tentative() {
		let profiling_frame = ProfilingFrame::new("single_tentative_detection");
		let mut tracker = ObjectTracker::new(vec!["object".to_string()], &profiling_frame);

		let output = update_after(&mut tracker, 1.0 / HIGH_HZ, vec![detection(0.5)]);
		let tracking_id = *tracker
			.track_validations
			.keys()
			.next()
			.expect("ByteTrack should expose a tentative ID");

		println!(
			"single_tentative_detection\tid={tracking_id}\tstate={:?}",
			validation_state(&tracker, tracking_id)
		);
		assert!(output.is_empty());
		assert!(matches!(
			validation_state(&tracker, tracking_id),
			TrackValidationState::Tentative { .. }
		));
	}

	#[test]
	fn first_zero_and_one_nanosecond_updates_are_well_defined() {
		let profiling_frame = ProfilingFrame::new("tiny_elapsed_updates");
		let mut tracker = ObjectTracker::new(vec!["object".to_string()], &profiling_frame);
		let first_update = Instant::now();

		assert!(
			tracker
				.update_at(vec![detection(0.5)], first_update)
				.is_empty()
		);
		let tracking_id = *tracker
			.track_validations
			.keys()
			.next()
			.expect("first native update should create a tentative ID");
		assert_eq!(tentative_visible_seconds(&tracker, tracking_id), 0.0);

		assert!(
			tracker
				.update_at(vec![detection(0.5)], first_update)
				.is_empty()
		);
		assert_eq!(tentative_visible_seconds(&tracker, tracking_id), 0.0);

		assert!(
			tracker
				.update_at(vec![detection(0.5)], first_update + Duration::from_nanos(1),)
				.is_empty()
		);
		let evidence = tentative_visible_seconds(&tracker, tracking_id);
		assert!(evidence.is_finite());
		assert!(evidence > 0.0 && evidence < 1e-8);
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

	#[test]
	fn lower_confidence_needs_more_reliable_visible_time() {
		let high_confidence_confirmation = confirmation_time_at(HIGH_HZ, CONFIDENCE);
		let lower_confidence_confirmation = confirmation_time_at(HIGH_HZ, 0.65);

		println!(
			"confidence_weighted_validation\tconfidence_0.90_s={high_confidence_confirmation:.3}\tconfidence_0.65_s={lower_confidence_confirmation:.3}"
		);
		assert!(lower_confidence_confirmation > high_confidence_confirmation);
	}

	#[test]
	fn continuous_reliable_detection_transitions_to_confirmed() {
		let profiling_frame = ProfilingFrame::new("continuous_confirmed");
		let mut tracker = ObjectTracker::new(vec!["object".to_string()], &profiling_frame);
		let tracking_id = warm_visible_track(&mut tracker, HIGH_HZ);

		assert_eq!(
			validation_state(&tracker, tracking_id),
			TrackValidationState::Confirmed
		);
	}

	#[test]
	fn reset_starts_a_fresh_epoch_without_reloading_detector_state() {
		let profiling_frame = ProfilingFrame::new("tracker_reset");
		let mut tracker = ObjectTracker::new(vec!["object".to_string()], &profiling_frame);
		let old_id = warm_visible_track(&mut tracker, HIGH_HZ);

		assert_eq!(
			validation_state(&tracker, old_id),
			TrackValidationState::Confirmed
		);
		assert!(tracker.last_update.is_some());
		assert!(tracker.update_number > 0);
		let pause_at = tracker.last_update.unwrap() + Duration::from_secs(12);
		assert!(tracker.update_at(vec![detection(0.5)], pause_at).is_empty());
		let replacement_id = visible_id_after_detections(&mut tracker, HIGH_HZ, 12);
		assert_ne!(replacement_id, old_id);

		tracker.reset();

		assert!(tracker.track_validations.is_empty());
		assert!(tracker.last_update.is_none());
		assert_eq!(tracker.update_number, 0);

		// BYTETracker itself was replaced, so its ID counter and active/lost
		// tracks are fresh as well. The first post-reset detection is tentative.
		assert!(
			tracker
				.update_at(vec![detection(0.5)], Instant::now())
				.is_empty()
		);
		let new_id = *tracker
			.track_validations
			.keys()
			.next()
			.expect("first post-reset detection should create a validation state");
		assert_eq!(new_id, 1);
		assert_eq!(tracker.labels, vec!["object".to_string()]);
		assert_eq!(tentative_visible_seconds(&tracker, new_id), 0.0);
		assert!(matches!(
			validation_state(&tracker, new_id),
			TrackValidationState::Tentative { .. }
		));
	}

	#[test]
	fn tentative_track_does_not_credit_an_unobserved_gap() {
		let profiling_frame = ProfilingFrame::new("tentative_gap");
		let mut tracker = ObjectTracker::new(vec!["object".to_string()], &profiling_frame);
		let interval_seconds = 1.0 / HIGH_HZ;

		for _ in 0..3 {
			assert!(update_after(&mut tracker, interval_seconds, vec![detection(0.5)]).is_empty());
		}
		let tracking_id = *tracker
			.track_validations
			.keys()
			.next()
			.expect("tentative ID should exist");
		let before_gap = tentative_visible_seconds(&tracker, tracking_id);

		assert!(update_after(&mut tracker, interval_seconds, Vec::new()).is_empty());
		assert!(update_after(&mut tracker, interval_seconds, vec![detection(0.5)]).is_empty());
		let after_reappearance = tentative_visible_seconds(&tracker, tracking_id);

		println!(
			"tentative_gap\tid={tracking_id}\tbefore={before_gap:.3}\tafter_reappearance={after_reappearance:.3}"
		);
		assert!((after_reappearance - before_gap).abs() < 0.001);

		let confirmed_id = visible_id_after_detections(&mut tracker, HIGH_HZ, 12);
		assert_eq!(confirmed_id, tracking_id);
		assert_eq!(
			validation_state(&tracker, tracking_id),
			TrackValidationState::Confirmed
		);
	}

	#[test]
	fn tentative_evidence_pauses_and_accumulates_across_short_bursts() {
		let profiling_frame = ProfilingFrame::new("tentative_detection_bursts");
		let mut tracker = ObjectTracker::new(vec!["object".to_string()], &profiling_frame);
		let interval_seconds = 1.0 / HIGH_HZ;
		let mut tracking_id = None;
		let mut evidence_after_bursts = Vec::new();

		for burst_index in 0..5 {
			for _ in 0..3 {
				let output = update_after(&mut tracker, interval_seconds, vec![detection(0.5)]);
				if let Some(object) = output.first() {
					tracking_id = Some(object.tracking_id);
				}
			}
			let id = tracking_id.unwrap_or_else(|| {
				*tracker
					.track_validations
					.keys()
					.next()
					.expect("tentative ID should exist")
			});
			tracking_id = Some(id);

			if validation_state(&tracker, id) == TrackValidationState::Confirmed {
				assert_eq!(burst_index, 3);
				break;
			}

			let before_gap = tentative_visible_seconds(&tracker, id);
			assert!(update_after(&mut tracker, interval_seconds, Vec::new()).is_empty());
			let after_gap = tentative_visible_seconds(&tracker, id);
			assert!((after_gap - before_gap).abs() < 0.001);
			evidence_after_bursts.push(after_gap);
		}

		let tracking_id = tracking_id.expect("the burst sequence should create a track");
		println!(
			"tentative_detection_bursts\tid={tracking_id}\tevidence_after_pauses={evidence_after_bursts:?}\tfinal_state={:?}",
			validation_state(&tracker, tracking_id)
		);
		assert_eq!(
			validation_state(&tracker, tracking_id),
			TrackValidationState::Confirmed
		);
		assert_eq!(tracker.track_validations.len(), 1);
	}

	#[test]
	fn tentative_track_does_not_treat_a_reusable_long_gap_as_visible_time() {
		let profiling_frame = ProfilingFrame::new("tentative_long_gap");
		let mut tracker = ObjectTracker::new(vec!["object".to_string()], &profiling_frame);

		assert!(update_after(&mut tracker, 1.0 / HIGH_HZ, vec![detection(0.5)]).is_empty());
		let tracking_id = *tracker
			.track_validations
			.keys()
			.next()
			.expect("tentative ID should exist");
		assert!(update_after(&mut tracker, 5.0, vec![detection(0.5)]).is_empty());

		println!(
			"tentative_long_gap\tid={tracking_id}\tstate={:?}",
			validation_state(&tracker, tracking_id)
		);
		assert_eq!(tentative_visible_seconds(&tracker, tracking_id), 0.0);
	}

	#[test]
	fn cadence_drop_does_not_confirm_a_young_track_early() {
		let profiling_frame = ProfilingFrame::new("young_track_15_to_3");
		let mut tracker = ObjectTracker::new(vec!["object".to_string()], &profiling_frame);

		assert!(update_after(&mut tracker, 1.0 / HIGH_HZ, vec![detection(0.5)]).is_empty());
		assert!(
			update_after(&mut tracker, 1.0 / LOW_HZ, vec![detection(0.5)]).is_empty(),
			"the 15 Hz contribution plus one 3 Hz observation is still below 0.45 confidence-seconds"
		);
		let output = update_after(&mut tracker, 1.0 / LOW_HZ, vec![detection(0.5)]);

		assert_eq!(output.len(), 1);
		assert_eq!(
			validation_state(&tracker, output[0].tracking_id),
			TrackValidationState::Confirmed
		);
	}

	#[test]
	fn stale_validation_states_are_cleaned_up_after_the_nominal_track_lifetime() {
		let profiling_frame = ProfilingFrame::new("validation_cleanup");
		let mut tracker = ObjectTracker::new(vec!["object".to_string()], &profiling_frame);
		let mut output = Vec::new();
		for _ in 0..12 {
			output = update_after(
				&mut tracker,
				1.0 / HIGH_HZ,
				vec![detection(0.25), detection(0.75)],
			);
		}
		assert_eq!(output.len(), 2);

		let survivor = output[0].clone();
		let stale_id = output
			.iter()
			.find(|object| object.tracking_id != survivor.tracking_id)
			.expect("second track should exist")
			.tracking_id;
		tracker
			.track_validations
			.get_mut(&stale_id)
			.expect("stale state should exist")
			.last_seen = Instant::now()
			- Duration::from_secs_f64(BYTETracker::DEFAULT_MAX_TRACKING_TIME_SECONDS + 0.1);

		update_after(
			&mut tracker,
			1.0 / HIGH_HZ,
			vec![detection(survivor.object.bbox.center_x)],
		);

		println!(
			"validation_cleanup\tsurvivor_id={}\tstale_id={}\tremaining_states={}",
			survivor.tracking_id,
			stale_id,
			tracker.track_validations.len()
		);
		assert!(
			tracker
				.track_validations
				.contains_key(&survivor.tracking_id)
		);
		assert!(!tracker.track_validations.contains_key(&stale_id));
	}

	#[test]
	fn multiple_track_ids_confirm_independently() {
		let profiling_frame = ProfilingFrame::new("multiple_validation_states");
		let mut tracker = ObjectTracker::new(vec!["object".to_string()], &profiling_frame);
		let mut output = Vec::new();
		for _ in 0..12 {
			output = update_after(
				&mut tracker,
				1.0 / HIGH_HZ,
				vec![detection(0.25), detection(0.75)],
			);
		}

		let ids = output
			.iter()
			.map(|object| object.tracking_id)
			.collect::<BTreeSet<_>>();
		println!("multiple_validation_states\tids={ids:?}");
		assert_eq!(ids.len(), 2);
		assert_eq!(tracker.track_validations.len(), 2);
		for tracking_id in ids {
			assert_eq!(
				validation_state(&tracker, tracking_id),
				TrackValidationState::Confirmed
			);
		}
	}

	#[test]
	fn long_pause_expires_track_and_validation_before_reassociation() {
		let profiling_frame = ProfilingFrame::new("wall_clock_pause");
		let mut tracker = ObjectTracker::new(vec!["object".to_string()], &profiling_frame);
		let mut last_id = None;

		for _ in 0..12 {
			let output = update_after(&mut tracker, 1.0 / HIGH_HZ, vec![detection(0.5)]);
			if let Some(object) = output.first() {
				last_id = Some(object.tracking_id);
			}
		}
		let before_pause = last_id.expect("track should be visible before the pause");
		let after_pause = update_after(&mut tracker, 12.0, vec![detection(0.5)]);

		assert!(
			after_pause.is_empty(),
			"the replacement track still needs native and EyeAI confirmation"
		);
		assert!(!tracker.track_validations.contains_key(&before_pause));

		let after_reconfirmation = visible_id_after_detections(&mut tracker, HIGH_HZ, 12);
		assert_ne!(after_reconfirmation, before_pause);
		assert_eq!(tracker.track_validations.len(), 1);
		assert_eq!(
			validation_state(&tracker, after_reconfirmation),
			TrackValidationState::Confirmed
		);
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

	#[test]
	fn cadence_drop_does_not_expire_a_lost_track_before_real_timeout() {
		let profiling_frame = ProfilingFrame::new("lost_track_rate_change");
		let mut tracker = ObjectTracker::new(vec!["object".to_string()], &profiling_frame);
		let original_id = warm_visible_track(&mut tracker, HIGH_HZ);

		// Four seconds at 15 Hz followed by two 3 Hz updates are only about 4.67
		// real seconds. Changing cadence must not change the ten-second lifetime.
		for _ in 0..60 {
			update_after(&mut tracker, 1.0 / HIGH_HZ, Vec::new());
		}
		update_after(&mut tracker, 1.0 / LOW_HZ, Vec::new());
		update_after(&mut tracker, 1.0 / LOW_HZ, Vec::new());
		let reacquired_id = visible_id_after_detections(&mut tracker, LOW_HZ, 4);

		println!(
			"lost_track_high_to_low\toriginal_id={original_id}\treacquired_id={reacquired_id}\treal_lost_s=4.667"
		);
		assert_eq!(
			reacquired_id, original_id,
			"the track must remain reusable before ten real seconds",
		);
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

	#[test]
	fn lost_track_lifetime_is_about_ten_real_seconds_at_constant_cadence() {
		let high_nine = id_after_constant_rate_loss(HIGH_HZ, 9.0);
		let low_nine = id_after_constant_rate_loss(LOW_HZ, 9.0);
		let high_ten = id_after_constant_rate_loss(HIGH_HZ, 10.0);
		let low_ten = id_after_constant_rate_loss(LOW_HZ, 10.0);
		let high_eleven = id_after_constant_rate_loss(HIGH_HZ, 11.0);
		let low_eleven = id_after_constant_rate_loss(LOW_HZ, 11.0);

		println!(
			"lost_track_constant_rates\thigh_9s={high_nine:?}\tlow_9s={low_nine:?}\thigh_10s={high_ten:?}\tlow_10s={low_ten:?}\thigh_11s={high_eleven:?}\tlow_11s={low_eleven:?}"
		);
		assert_eq!(high_nine.0, high_nine.1);
		assert_eq!(low_nine.0, low_nine.1);
		assert_eq!(high_ten.0, high_ten.1);
		assert_eq!(low_ten.0, low_ten.1);
		assert_ne!(high_eleven.0, high_eleven.1);
		assert_ne!(low_eleven.0, low_eleven.1);
	}

	#[test]
	fn lost_track_lifetime_uses_real_time_for_all_cadence_schedules() {
		for lost_seconds in [9_u64, 10, 11] {
			let total = Duration::from_secs(lost_seconds);
			let mut schedules = vec![
				("single_pause", vec![total]),
				("constant_15", segmented_loss_intervals(total, &[], HIGH_HZ)),
				("constant_3", segmented_loss_intervals(total, &[], LOW_HZ)),
				(
					"15_to_3",
					segmented_loss_intervals(total, &[(Duration::from_secs(4), HIGH_HZ)], LOW_HZ),
				),
				(
					"3_to_15",
					segmented_loss_intervals(total, &[(Duration::from_secs(4), LOW_HZ)], HIGH_HZ),
				),
				(
					"10_to_3_to_10",
					segmented_loss_intervals(
						total,
						&[
							(Duration::from_secs(3), NORMAL_HZ),
							(Duration::from_secs(3), LOW_HZ),
						],
						NORMAL_HZ,
					),
				),
			];
			schedules.push(("irregular", irregular_loss_intervals(total)));

			for (name, intervals) in schedules {
				assert_eq!(intervals.iter().sum::<Duration>(), total);
				let (original_id, reacquired_id, immediate_reassociation) =
					id_after_loss_intervals(&intervals);
				println!(
					"real_time_lifetime\tschedule={name}\tlost_s={lost_seconds}\tupdates={}\toriginal_id={original_id}\treacquired_id={reacquired_id}\timmediate_same_id={immediate_reassociation}",
					intervals.len()
				);

				if lost_seconds <= 10 {
					assert_eq!(reacquired_id, original_id, "{name} at {lost_seconds}s");
					assert!(immediate_reassociation, "{name} at {lost_seconds}s");
				} else {
					assert_ne!(reacquired_id, original_id, "{name} at {lost_seconds}s");
					assert!(!immediate_reassociation, "{name} at {lost_seconds}s");
				}
			}
		}
	}

	#[test]
	fn consecutive_detection_failures_are_reassociated_at_both_cadences() {
		for (name, hz, failure_count) in [
			("high_15hz", HIGH_HZ, 4_usize),
			("low_3hz", LOW_HZ, 3_usize),
		] {
			let profiling_frame = ProfilingFrame::new(format!("consecutive_failures_{name}"));
			let mut tracker = ObjectTracker::new(vec!["object".to_string()], &profiling_frame);
			let original_id = warm_visible_track(&mut tracker, hz);
			let mut failure_outputs = Vec::new();
			for _ in 0..failure_count {
				failure_outputs.push(update_after(&mut tracker, 1.0 / hz, Vec::new()).len());
			}
			let reacquired_id = visible_id_after_detections(&mut tracker, hz, 12);
			println!(
				"consecutive_detection_failures\tname={name}\thz={hz}\tfailures={failure_count}\tfailure_outputs={failure_outputs:?}\toriginal_id={original_id}\treacquired_id={reacquired_id}"
			);
			assert_eq!(
				reacquired_id, original_id,
				"short consecutive detection failure should not force an ID change at {name}"
			);
		}
	}

	#[test]
	fn multi_object_crossing_constant_vs_15_to_3_to_15() {
		let constant_intervals = repeated_interval(HIGH_HZ, 60);
		let mut variable_intervals = repeated_interval(HIGH_HZ, 20);
		variable_intervals.extend(repeated_interval(LOW_HZ, 4));
		variable_intervals.extend(repeated_interval(HIGH_HZ, 20));

		let constant = run_multi_scenario(
			"multi_crossing_constant_15hz",
			&constant_intervals,
			crossing_truths,
		);
		let variable = run_multi_scenario(
			"multi_crossing_15_to_3_to_15",
			&variable_intervals,
			crossing_truths,
		);
		print_multi_metrics(&constant);
		print_multi_metrics(&variable);

		for metrics in [&constant, &variable] {
			assert_eq!(metrics.visible_by_truth.len(), 2);
			assert!(metrics.visible_by_truth.iter().all(|count| *count > 0));
			assert_eq!(metrics.post_visible_misses_by_truth, [0, 0]);
			assert_eq!(metrics.duplicate_id_updates, 0);
			assert!(metrics.max_center_error.is_finite());
		}
		assert_eq!(variable.id_switches, constant.id_switches);
		assert!(variable.max_center_error < 0.01);
	}

	#[test]
	fn multi_object_close_then_separate_constant_vs_15_to_3_to_15() {
		let constant_intervals = repeated_interval(HIGH_HZ, 60);
		let mut variable_intervals = repeated_interval(HIGH_HZ, 20);
		variable_intervals.extend(repeated_interval(LOW_HZ, 4));
		variable_intervals.extend(repeated_interval(HIGH_HZ, 20));

		let constant = run_multi_scenario(
			"multi_close_separate_constant_15hz",
			&constant_intervals,
			close_then_separate_truths,
		);
		let variable = run_multi_scenario(
			"multi_close_separate_15_to_3_to_15",
			&variable_intervals,
			close_then_separate_truths,
		);
		print_multi_metrics(&constant);
		print_multi_metrics(&variable);

		for metrics in [&constant, &variable] {
			assert_eq!(metrics.visible_by_truth.len(), 2);
			assert!(metrics.visible_by_truth.iter().all(|count| *count > 0));
			assert_eq!(metrics.post_visible_misses_by_truth, [0, 0]);
			assert_eq!(metrics.id_switches, 0);
			assert_eq!(metrics.duplicate_id_updates, 0);
			assert!(metrics.max_center_error.is_finite());
		}
	}
}
