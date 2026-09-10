use super::*;

#[test]
fn blocked_create_across_stop_is_discarded_and_threads_are_joined() {
	let registry = Arc::new(AudioSessions::<Engine>::default());
	let id = registry.begin();
	let session = registry.get(id).unwrap();
	let live = Arc::new(AtomicUsize::new(0));
	let (entered_tx, entered_rx) = mpsc::channel();
	let (release_tx, release_rx) = mpsc::channel();
	let worker_session = session.clone();
	let worker_live = live.clone();
	let old = thread::spawn(move || {
		worker_session
			.change(
				|| {
					let candidate = Engine::new(&worker_session, &worker_live);
					entered_tx.send(()).unwrap();
					release_rx.recv_timeout(TIMEOUT).unwrap();
					Ok::<_, ()>(candidate)
				},
				|_| panic!("stopped create must not update"),
			)
			.unwrap();
	});
	entered_rx.recv_timeout(TIMEOUT).unwrap();
	registry.invalidate(id); // Must return while A owns its engine lock.
	assert!(!session.is_active());
	assert_eq!(registry.0.lock().unwrap().active_id, None);
	let cleanup_registry = registry.clone();
	let cleanup = thread::spawn(move || cleanup_registry.destroy(id));
	release_tx.send(()).unwrap();
	old.join().unwrap();
	cleanup.join().unwrap();
	assert!(session.engine.lock().unwrap().is_none());
	assert!(registry.0.lock().unwrap().sessions.is_empty());
	assert_eq!(live.load(Ordering::SeqCst), 0);
}

#[test]
fn active_missing_engine_and_disconnect_recover_without_changing_identity() {
	let registry = AudioSessions::<Engine>::default();
	let id = registry.begin();
	let session = registry.get(id).unwrap();
	let live = Arc::new(AtomicUsize::new(0));
	let mut creations = 0;
	session
		.change(
			|| {
				creations += 1;
				Ok::<_, ()>(Engine::new(&session, &live))
			},
			|engine| {
				engine.updates += 1;
				true
			},
		)
		.unwrap();
	assert_eq!(creations, 2); // initial missing engine, then legitimate reconnect
	assert_eq!(registry.0.lock().unwrap().active_id, Some(id));
	create(&session, &live); // repeated create is idempotent
	assert_eq!(live.load(Ordering::SeqCst), 1);
	registry.invalidate(id);
	registry.invalidate(id);
	registry.destroy(id);
	registry.destroy(id);
	create(&session, &live); // retained old handle cannot resurrect a session
	assert_eq!(live.load(Ordering::SeqCst), 0);
}

#[test]
fn failures_and_panics_still_allow_final_destroy() {
	let registry = AudioSessions::<Engine>::default();
	let id = registry.begin();
	let session = registry.get(id).unwrap();
	let live = Arc::new(AtomicUsize::new(0));
	assert!(
		session
			.change(|| Err::<Engine, _>("create failed"), |_| false)
			.is_err()
	);
	create(&session, &live);
	assert!(
		session
			.change(|| Err::<Engine, _>("recreate failed"), |_| true)
			.is_err()
	);
	assert_eq!(live.load(Ordering::SeqCst), 0);
	create(&session, &live);
	assert!(
		catch_unwind(AssertUnwindSafe(|| {
			session
				.change(
					|| Ok::<_, ()>(Engine::new(&session, &live)),
					|_| panic!("send failed"),
				)
				.unwrap();
		}))
		.is_err()
	);
	registry.destroy(id); // recovers the poisoned per-session engine lock
	assert_eq!(live.load(Ordering::SeqCst), 0);
}
