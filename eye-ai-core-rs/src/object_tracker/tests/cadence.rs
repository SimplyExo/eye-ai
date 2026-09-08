use super::*;

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
	irregular_intervals.extend([1.0 / 10.0, 1.0 / 4.0, 1.0 / 15.0, 1.0 / 3.0, 1.0 / 8.0].repeat(3));
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
		// At 3 Hz, 0.6 would move a 0.2-wide normalized box by its full
		// width per update (continuous IoU = 0). Keep this regression inside
		// the actual 0.8 association-cost geometry instead of relying on the
		// old pixel-style +1 IoU behavior.
		run_scenario("M_linear_constant_low", &b_intervals, &b_present, 0.3),
		run_scenario(
			"N_linear_3_to_15",
			&low_to_high_intervals,
			&low_to_high_present,
			0.3,
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
