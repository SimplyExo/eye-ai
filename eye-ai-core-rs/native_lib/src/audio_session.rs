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

	pub(crate) fn invalidate(&self, id: u64) {
		let mut registry = self.0.lock().unwrap();
		if let Some(session) = registry.sessions.get(&id) {
			session.invalidate();
		}
		if registry.active_id == Some(id) {
			registry.active_id = None;
		}
	}

	pub(crate) fn destroy(&self, id: u64) {
		self.invalidate(id);
		let session = self.0.lock().unwrap().sessions.remove(&id);
		if let Some(session) = session {
			session.destroy();
		}
	}
}

#[cfg(test)]
#[path = "audio_session/tests/mod.rs"]
mod tests;
