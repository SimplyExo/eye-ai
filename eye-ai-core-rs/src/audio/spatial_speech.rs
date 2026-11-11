//! One bounded, cancellable speech source sharing the spatial audio context.
use alto::{Context, Mono, Source, SourceState};
use std::sync::{
	Arc,
	atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

pub enum PlaybackEvent {
	Started,
	Ended(bool),
}

#[derive(Default)]
pub struct SpatialSpeech {
	cancelled: Arc<AtomicBool>,
	worker: Option<std::thread::JoinHandle<()>>,
}
impl SpatialSpeech {
	pub fn stop(&mut self) {
		self.cancelled.store(true, Ordering::Release);
		if let Some(worker) = self.worker.take() {
			let _ = worker.join();
		}
	}
	pub fn play(
		&mut self,
		context: Arc<Context>,
		pcm: Vec<u8>,
		rate: i32,
		position: [f32; 3],
		gain: f32,
		mut on_event: impl FnMut(PlaybackEvent) + Send + 'static,
	) -> Result<(), String> {
		if pcm.is_empty()
			|| pcm.len() % 2 != 0
			|| !(8000..=192000).contains(&rate)
			|| position.iter().any(|v| !v.is_finite())
			|| !gain.is_finite()
		{
			return Err("Invalid speech PCM or position".into());
		}
		self.stop();
		self.cancelled.store(false, Ordering::Release);
		let cancelled = self.cancelled.clone();
		self.worker = Some(
			std::thread::Builder::new()
				.name("Spatial Speech".into())
				.spawn(move || {
					let result = (|| -> Result<(), alto::AltoError> {
						let mut source = context.new_static_source()?;
						source.set_rolloff_factor(0.0)?;
						source.set_gain(gain.clamp(0.75, 1.0))?;
						source.set_position(position)?;
						let samples: Vec<Mono<i16>> = pcm
							.chunks_exact(2)
							.map(|bytes| Mono {
								center: i16::from_le_bytes([bytes[0], bytes[1]]),
							})
							.collect();
						drop(pcm);
						source.set_buffer(Arc::new(context.new_buffer(&samples, rate)?))?;
						if cancelled.load(Ordering::Acquire) {
							return Ok(());
						}
						source.play();
						on_event(PlaybackEvent::Started);
						while source.state() == SourceState::Playing
							&& !cancelled.load(Ordering::Acquire)
						{
							std::thread::sleep(Duration::from_millis(10));
						}
						source.stop();
						source.clear_buffer();
						Ok(())
					})();
					on_event(PlaybackEvent::Ended(
						result.is_ok() && !cancelled.load(Ordering::Acquire),
					));
				})
				.map_err(|err| err.to_string())?,
		);
		Ok(())
	}
}
impl Drop for SpatialSpeech {
	fn drop(&mut self) {
		self.stop();
	}
}
