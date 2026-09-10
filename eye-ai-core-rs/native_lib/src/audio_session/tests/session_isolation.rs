use super::*;

#[test]
fn new_session_runs_while_old_create_and_finalization_are_blocked() {
	let registry = Arc::new(AudioSessions::<Engine>::default());
	let a = registry.begin();
	let old_session = registry.get(a).unwrap();
	let live = Arc::new(AtomicUsize::new(0));
	let (entered_tx, entered_rx) = mpsc::channel();
	let (release_tx, release_rx) = mpsc::channel();
	let old_live = live.clone();
	let old = thread::spawn(move || {
		old_session
			.change(
				|| {
					entered_tx.send(()).unwrap();
					release_rx.recv_timeout(TIMEOUT).unwrap();
					Ok::<_, ()>(Engine::new(&old_session, &old_live))
				},
				|_| panic!("A was invalidated"),
			)
			.unwrap();
	});
	entered_rx.recv_timeout(TIMEOUT).unwrap();
	registry.invalidate(a);
	let cleanup_registry = registry.clone();
	let cleanup = thread::spawn(move || cleanup_registry.destroy(a));
	let b = registry.begin();
	let new_session = registry.get(b).unwrap();
	create(&new_session, &live);
	assert!(new_session.engine.lock().unwrap().is_some());
	release_tx.send(()).unwrap();
	old.join().unwrap();
	cleanup.join().unwrap();
	registry.destroy(a); // Even a second late finalizer cannot remove B.
	assert_eq!(registry.0.lock().unwrap().active_id, Some(b));
	assert!(new_session.is_active());
	assert_eq!(live.load(Ordering::SeqCst), 1);
	registry.destroy(b);
	assert_eq!(live.load(Ordering::SeqCst), 0);
}

#[test]
fn blocked_send_cannot_mutate_or_recreate_in_new_session() {
	let registry = AudioSessions::<Engine>::default();
	let a = registry.begin();
	let old_session = registry.get(a).unwrap();
	let live = Arc::new(AtomicUsize::new(0));
	create(&old_session, &live);
	let (entered_tx, entered_rx) = mpsc::channel();
	let (release_tx, release_rx) = mpsc::channel();
	let old = thread::spawn(move || {
		old_session
			.change(
				|| -> Result<Engine, ()> { panic!("old recovery forbidden") },
				|engine| {
					entered_tx.send(()).unwrap();
					release_rx.recv_timeout(TIMEOUT).unwrap();
					engine.updates += 1; // A may finish only into A's retired engine.
					true // Old device requests recovery AFTER stop.
				},
			)
			.unwrap();
	});
	entered_rx.recv_timeout(TIMEOUT).unwrap();
	registry.invalidate(a);
	let b = registry.begin();
	let new_session = registry.get(b).unwrap();
	create(&new_session, &live);
	release_tx.send(()).unwrap();
	old.join().unwrap();
	registry.destroy(a);
	assert_eq!(
		new_session.engine.lock().unwrap().as_ref().unwrap().updates,
		0
	);
	assert!(new_session.is_active());
	assert_eq!(live.load(Ordering::SeqCst), 1);
	registry.destroy(b);
	assert_eq!(live.load(Ordering::SeqCst), 0);
}

#[test]
fn rapid_cycles_and_old_settings_never_touch_new_sessions() {
	let registry = AudioSessions::<Engine>::default();
	let live = Arc::new(AtomicUsize::new(0));
	for _ in 0..50 {
		let a = registry.begin();
		let old = registry.get(a).unwrap();
		create(&old, &live);
		let b = registry.begin();
		assert!(b > a);
		assert!(!old.is_active());
		let new = registry.get(b).unwrap();
		old.settings.write().unwrap().frequency = 123.0;
		assert_ne!(new.settings.read().unwrap().frequency, 123.0);
		registry.destroy(a);
		registry.destroy(b); // no engine was ever created for B
		assert!(registry.0.lock().unwrap().sessions.is_empty());
	}
	assert_eq!(live.load(Ordering::SeqCst), 0);
}
