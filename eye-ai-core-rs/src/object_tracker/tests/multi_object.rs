use super::*;

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
