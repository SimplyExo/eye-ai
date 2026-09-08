use super::*;

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
