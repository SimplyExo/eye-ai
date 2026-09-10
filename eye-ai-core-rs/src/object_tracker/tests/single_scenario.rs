use super::*;
use std::collections::BTreeSet;

#[derive(Debug)]
pub(super) struct ScenarioMetrics {
	pub name: &'static str,
	pub updates: usize,
	pub detection_updates: usize,
	pub visible_updates: usize,
	pub post_visible_misses_with_detection: usize,
	pub max_output_count: usize,
	pub unique_ids: BTreeSet<i32>,
	pub id_switches: usize,
	pub first_visible_seconds: Option<f32>,
	pub max_center_error: f32,
}

pub(super) fn run_scenario(
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

pub(super) fn print_metrics(metrics: &ScenarioMetrics) {
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
