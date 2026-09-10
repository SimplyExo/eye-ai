use super::*;
use std::collections::BTreeSet;

const MULTI_BOX_WIDTH: f32 = 0.16;
const MULTI_BOX_HEIGHT: f32 = 0.16;
const MAX_ASSIGNMENT_ERROR: f32 = 0.30;

#[derive(Debug)]
pub(super) struct MultiScenarioMetrics {
	pub name: &'static str,
	pub updates: usize,
	pub visible_truth_updates: usize,
	pub visible_by_truth: Vec<usize>,
	pub post_visible_misses_by_truth: Vec<usize>,
	pub reassociations_same_id: usize,
	pub max_output_count: usize,
	pub duplicate_id_updates: usize,
	pub unique_ids: BTreeSet<i32>,
	pub id_switches: usize,
	pub id_switches_by_truth: Vec<usize>,
	pub first_ids: Vec<Option<i32>>,
	pub last_ids: Vec<Option<i32>>,
	pub first_visible_seconds: Vec<Option<f32>>,
	pub max_center_error: f32,
}

fn center_error(output: &TrackedObject, truth: TruthBox) -> f32 {
	(output.object.bbox.center_x - truth.center_x)
		.hypot(output.object.bbox.center_y - truth.center_y)
}

fn assign_outputs(output: &[TrackedObject], truths: &[TruthBox]) -> Vec<Option<usize>> {
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

pub(super) fn run_multi_scenario<F>(
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

		let assignments = assign_outputs(&output, &truths);
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

pub(super) fn print_multi_metrics(metrics: &MultiScenarioMetrics) {
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

pub(super) fn crossing_truths(elapsed_seconds: f32) -> Vec<TruthBox> {
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

pub(super) fn close_then_separate_truths(elapsed_seconds: f32) -> Vec<TruthBox> {
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
