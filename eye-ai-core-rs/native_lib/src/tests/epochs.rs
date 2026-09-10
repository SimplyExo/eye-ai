use super::super::*;
use std::{sync::mpsc, thread, time::Duration};

fn detection() -> DetectedObject {
	DetectedObject::new(
		"person".to_owned(),
		0,
		1.0,
		BoundingBox::new(0.5, 0.5, 0.2, 0.2),
	)
}

#[test]
fn reset_seam_waits_for_old_native_mutation_and_clears_confirmed_evidence() {
	assert!(YOLO_MODEL.read().unwrap().is_none());
	*OBJECT_TRACKER.lock().unwrap() = Some(ObjectTracker::new(
		vec!["person".to_owned()],
		&OBJECT_PROFILING_FRAME,
	));
	let mut confirmed = vec![];
	for _ in 0..5 {
		confirmed = OBJECT_TRACKER
			.lock()
			.unwrap()
			.as_mut()
			.unwrap()
			.update(vec![detection()]);
		thread::sleep(Duration::from_millis(150));
	}
	assert_eq!(confirmed.len(), 1);
	assert_eq!(confirmed[0].tracking_id, 1);

	let (entered_tx, entered_rx) = mpsc::channel();
	let (release_tx, release_rx) = mpsc::channel();
	let old = thread::spawn(move || {
		let _model = YOLO_MODEL.write().unwrap();
		entered_tx.send(()).unwrap();
		release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
		let result = OBJECT_TRACKER
			.lock()
			.unwrap()
			.as_mut()
			.unwrap()
			.update(vec![detection()]);
		assert_eq!(result.len(), 1, "old completion still belongs to A");
	});
	entered_rx.recv_timeout(Duration::from_secs(3)).unwrap();
	let (requested_tx, requested_rx) = mpsc::channel();
	let (reset_tx, reset_rx) = mpsc::channel();
	let reset = thread::spawn(move || {
		requested_tx.send(()).unwrap();
		resetObjectTracker();
		reset_tx.send(()).unwrap();
	});
	requested_rx.recv_timeout(Duration::from_secs(3)).unwrap();
	assert!(matches!(
		reset_rx.recv_timeout(Duration::from_millis(100)),
		Err(mpsc::RecvTimeoutError::Timeout)
	));
	release_tx.send(()).unwrap();
	old.join().unwrap();
	reset_rx.recv_timeout(Duration::from_secs(3)).unwrap();
	reset.join().unwrap();

	let _model = YOLO_MODEL.write().unwrap();
	let mut tracker_slot = OBJECT_TRACKER.lock().unwrap();
	let first_b = tracker_slot.as_mut().unwrap().update(vec![detection()]);
	assert!(
		first_b.is_empty(),
		"B at the same position must be TENTATIVE"
	);
	assert!(_model.is_none(), "reset must never load a detector");
	*tracker_slot = None;
}
