//! Session identity is allocated before any device work. Registry operations never
//! wait for an engine lock; retired sessions remain owned until worker-side destroy.
use eye_ai_core_rs::audio::SpatialAudioSettings;
use std::{
	collections::HashMap,
	sync::{
		Arc, Mutex, RwLock,
		atomic::{AtomicBool, AtomicU64, Ordering},
	},
};

pub(crate) struct AudioSession<E> {
	pub(crate) active: Arc<AtomicBool>,
	pub(crate) object_audio_playback_epoch: Arc<AtomicU64>,
	pub(crate) settings: Arc<RwLock<SpatialAudioSettings>>,
	engine: Mutex<Option<E>>,
}

impl<E> AudioSession<E> {
	pub(crate) fn is_active(&self) -> bool {
		self.active.load(Ordering::Acquire)
	}

	fn invalidate(&self) {
		self.active.store(false, Ordering::Release);
	}

	/// All device construction, mutation and retirement is serial within this
	/// session only. A blocked A never holds a lock needed by stop or session B.
	pub(crate) fn change<Error>(
		&self,
		mut create: impl FnMut() -> Result<E, Error>,
		update: impl FnOnce(&mut E) -> bool,
	) -> Result<(), Error> {
		let mut engine = self.engine.lock().unwrap();
		if !self.is_active() {
			return Ok(());
		}
		if engine.is_none() {
			let candidate = create()?;
			if !self.is_active() {
				drop(candidate);
				return Ok(());
			}
			*engine = Some(candidate);
		}
		if self.is_active() && update(engine.as_mut().unwrap()) {
			// Device disconnect: retire this engine before replacing it. Both the
			// replacement and its playback threads carry the same session token.
			drop(engine.take());
			if self.is_active() {
				let candidate = create()?;
				if self.is_active() {
					*engine = Some(candidate);
				}
			}
		}
		Ok(())
	}

	fn destroy(&self) {
		self.invalidate();
		// A panic in device/update code must not prevent final resource cleanup.
		let engine = self.engine.lock().unwrap_or_else(|e| e.into_inner()).take();
		drop(engine);
	}
}

struct Registry<E> {
	next_id: u64,
	active_id: Option<u64>,
	sessions: HashMap<u64, Arc<AudioSession<E>>>,
}

pub(crate) struct AudioSessions<E>(Mutex<Registry<E>>);

impl<E> Default for AudioSessions<E> {
	fn default() -> Self {
		Self(Mutex::new(Registry {
			next_id: 0,
			active_id: None,
			sessions: HashMap::new(),
		}))
	}
}

impl<E> AudioSessions<E> {
	pub(crate) fn begin(&self) -> u64 {
		let mut registry = self.0.lock().unwrap();
		if let Some(old) = registry.active_id.and_then(|id| registry.sessions.get(&id)) {
			old.invalidate();
		}
		registry.next_id = registry
			.next_id
			.checked_add(1)
			.expect("audio session IDs exhausted");
		let id = registry.next_id;
		registry.sessions.insert(
			id,
			Arc::new(AudioSession {
				active: Arc::new(AtomicBool::new(true)),
				object_audio_playback_epoch: Arc::new(AtomicU64::new(0)),
				settings: Arc::new(RwLock::new(SpatialAudioSettings::default())),
				engine: Mutex::new(None),
			}),
		);
		registry.active_id = Some(id);
		id
	}

	pub(crate) fn get(&self, id: u64) -> Option<Arc<AudioSession<E>>> {
		self.0.lock().unwrap().sessions.get(&id).cloned()
	}

	/// Synchronous stop boundary: only a short registry lock and an atomic store.
	pub(crate) fn invalidate(&self, id: u64) {
		let mut registry = self.0.lock().unwrap();
		if let Some(session) = registry.sessions.get(&id) {
			session.invalidate();
		}
		if registry.active_id == Some(id) {
			registry.active_id = None;
		}
	}

	/// Worker only. Never removes or destroys another session's engine.
	pub(crate) fn destroy(&self, id: u64) {
		self.invalidate(id);
		let session = self.0.lock().unwrap().sessions.remove(&id);
		if let Some(session) = session {
			session.destroy();
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::{
		panic::{AssertUnwindSafe, catch_unwind},
		sync::{atomic::AtomicUsize, mpsc},
		thread,
		time::Duration,
	};

	const TIMEOUT: Duration = Duration::from_secs(5);

	struct Engine {
		updates: usize,
		running: Arc<AtomicBool>,
		thread: Option<thread::JoinHandle<()>>,
	}

	impl Engine {
		fn new(session: &AudioSession<Self>, live: &Arc<AtomicUsize>) -> Self {
			let active = session.active.clone();
			let running = Arc::new(AtomicBool::new(true));
			let thread_running = running.clone();
			let live = live.clone();
			live.fetch_add(1, Ordering::SeqCst);
			let thread = thread::spawn(move || {
				while active.load(Ordering::Acquire) && thread_running.load(Ordering::Acquire) {
					thread::park_timeout(Duration::from_millis(2));
				}
				live.fetch_sub(1, Ordering::SeqCst);
			});
			Self {
				updates: 0,
				running,
				thread: Some(thread),
			}
		}
	}

	impl Drop for Engine {
		fn drop(&mut self) {
			self.running.store(false, Ordering::Release);
			let thread = self.thread.take().unwrap();
			thread.thread().unpark();
			thread.join().unwrap();
		}
	}

	fn create(session: &AudioSession<Engine>, live: &Arc<AtomicUsize>) {
		session
			.change(|| Ok::<_, ()>(Engine::new(session, live)), |_| false)
			.unwrap();
	}

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
}
