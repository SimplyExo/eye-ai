use super::*;

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
			assert_eq!(burst_index, 4);
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
fn cadence_drop_does_not_confirm_a_young_track_early() {
	let profiling_frame = ProfilingFrame::new("young_track_15_to_3");
	let mut tracker = ObjectTracker::new(vec!["object".to_string()], &profiling_frame);

	assert!(update_after(&mut tracker, 1.0 / HIGH_HZ, vec![detection(0.5)]).is_empty());
	assert!(
		update_after(&mut tracker, 1.0 / LOW_HZ, vec![detection(0.5)]).is_empty(),
		"the 15 Hz contribution plus one 3 Hz observation is still below 0.5 confidence-seconds"
	);
	let output = update_after(&mut tracker, 1.0 / LOW_HZ, vec![detection(0.5)]);

	assert_eq!(output.len(), 1);
	assert_eq!(
		validation_state(&tracker, output[0].tracking_id),
		TrackValidationState::Confirmed
	);
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
