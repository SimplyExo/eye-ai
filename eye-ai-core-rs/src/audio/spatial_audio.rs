use alto::{
	Alto, Context, ContextAttrs, DeviceObject, DistanceModel, Mono, OutputDevice, Source,
	SourceState, ext::Alc,
};
use eye_ai_core_rs_profiling_attribute::profile_function;
use std::{
	collections::{HashMap, VecDeque},
	sync::{
		Arc, RwLock,
		atomic::{AtomicBool, AtomicU64, Ordering},
	},
	time::Duration,
};
use thiserror::Error;
use tracing::{debug, error, trace};
use tracing_tracy::client::secondary_frame_mark;

use crate::{
	ProfilingFrame, TrackedObject,
	audio::{
		CalculateSoundOrigin, DepthAudioSourceData, IVec2, ObjectAudioSourceData, ObjectLabelData,
		SpatialAudioContent, SpatialAudioSettings, Vec3, spatial_audio_content::AudioFileData,
	},
};

#[derive(Debug, Error)]
pub enum SpatialAudioError {
	#[error("Alto error: {0}")]
	AltoError(#[from] alto::AltoError),
	#[error("JSON error: {0}")]
	JsonError(#[from] json::Error),
	#[error("Audio thread error: {0}")]
	ThreadError(#[from] std::io::Error),
}

#[derive(Clone)]
struct PlaybackLifetime {
	running: Arc<AtomicBool>,
	session_active: Arc<AtomicBool>,
}

impl PlaybackLifetime {
	fn is_running(&self) -> bool {
		self.running.load(Ordering::Acquire) && self.session_active.load(Ordering::Acquire)
	}
}

struct QueuedObjectAudio {
	epoch: u64,
	source: ObjectAudioSourceData,
}

/// A transient empty snapshot only clears announcements that have not started yet.
/// Explicit lifecycle/generation invalidation advances the shared session epoch, which
/// also interrupts the currently playing announcement.
struct ObjectAudioQueue {
	pending: VecDeque<QueuedObjectAudio>,
	playback_epoch: Arc<AtomicU64>,
}

#[cfg(test)]
mod lifecycle_tests {
	use super::*;
	use std::time::Instant;

	#[test]
	#[ignore = "requires OpenAL Soft null backend: ALSOFT_DRIVERS=null cargo test --offline real_playback_threads -- --ignored"]
	fn real_playback_threads_exit_on_session_invalidation_and_engine_retirement() {
		let active = Arc::new(AtomicBool::new(true));
		let settings = Arc::new(RwLock::new(SpatialAudioSettings {
			depth_audio_paused: true,
			object_audio_paused: true,
			..Default::default()
		}));
		let content = Arc::new(SpatialAudioContent::new(
			AudioFileData {
				samples: vec![0; 48_000],
				sample_rate: 48_000,
			},
			HashMap::new(),
		));
		let profiling = Arc::new(ProfilingFrame::new_unretained("audio_lifecycle_test"));
		let create = || {
			SpatialAudio::new_in_session(
				settings.clone(),
				content.clone(),
				profiling.clone(),
				active.clone(),
				Arc::new(AtomicU64::new(0)),
			)
			.unwrap()
		};
		let first = create();
		drop(first); // Engine retirement must not invalidate the enclosing session.
		assert!(active.load(Ordering::Acquire));
		let second = create();
		active.store(false, Ordering::Release);
		// Keep the engine alive, as if send still owned it. Invalidation alone
		// must terminate BOTH real playback threads, including their paused paths.
		let deadline = Instant::now() + Duration::from_secs(3);
		while !second.depth_audio_thread.as_ref().unwrap().is_finished()
			|| !second.object_audio_thread.as_ref().unwrap().is_finished()
		{
			assert!(
				Instant::now() < deadline,
				"playback thread survived invalidation"
			);
			std::thread::sleep(Duration::from_millis(2));
		}
		drop(second);
		// A constructor that finishes after stop also inherits the revoked token.
		let late = create();
		drop(late);
	}
}

impl ObjectAudioQueue {
	fn new(playback_epoch: Arc<AtomicU64>) -> Self {
		Self {
			pending: VecDeque::new(),
			playback_epoch,
		}
	}

	fn update(&mut self, sources: VecDeque<ObjectAudioSourceData>) {
		if sources.is_empty() {
			self.pending.clear();
			return;
		}
		let epoch = self.playback_epoch.load(Ordering::Acquire);
		self.pending.retain(|old| {
			old.epoch == epoch
				&& sources
					.iter()
					.any(|new| old.source.object_id == new.object_id)
		});
		for source in sources {
			if let Some(old) = self
				.pending
				.iter_mut()
				.find(|old| old.source.object_id == source.object_id)
			{
				old.epoch = epoch;
				old.source = source;
			} else {
				while self.pending.len() >= 6 {
					self.pending.pop_front();
				}
				self.pending.push_back(QueuedObjectAudio { epoch, source });
			}
		}
	}

	fn pop(&mut self) -> Option<(u64, ObjectAudioSourceData)> {
		let current_epoch = self.playback_epoch.load(Ordering::Acquire);
		while let Some(queued) = self.pending.pop_front() {
			if queued.epoch == current_epoch {
				return Some((queued.epoch, queued.source));
			}
		}
		None
	}
}

pub struct SpatialAudio {
	_alto: Alto,
	_device: OutputDevice,
	_context: Arc<Context>,
	depth_audio_sources_data: Arc<RwLock<Vec<DepthAudioSourceData>>>,
	depth_audio_thread: Option<std::thread::JoinHandle<()>>,
	object_audio_sources_data: Arc<RwLock<ObjectAudioQueue>>,
	object_audio_thread: Option<std::thread::JoinHandle<()>>,
	lifetime: PlaybackLifetime,
	pub settings: Arc<RwLock<SpatialAudioSettings>>,
	content: Arc<SpatialAudioContent>,
	profiling_frame: Arc<ProfilingFrame>,
}
impl Drop for SpatialAudio {
	fn drop(&mut self) {
		self.lifetime.running.store(false, Ordering::Release);
		for thread in [&self.depth_audio_thread, &self.object_audio_thread]
			.into_iter()
			.flatten()
		{
			thread.thread().unpark();
		}
		// Native session destruction runs on its worker, never on the Android
		// main thread. Join before releasing the device/context; do not detach.
		for thread in [
			self.depth_audio_thread.take(),
			self.object_audio_thread.take(),
		]
		.into_iter()
		.flatten()
		{
			if thread.join().is_err() {
				error!("Audio playback thread panicked during its lifetime");
			}
		}
	}
}
impl SpatialAudio {
	#[profile_function("profiling_frame")]
	pub fn new(
		settings: Arc<RwLock<SpatialAudioSettings>>,
		content: Arc<SpatialAudioContent>,
		profiling_frame: Arc<ProfilingFrame>,
	) -> Result<Self, SpatialAudioError> {
		Self::new_in_session(
			settings,
			content,
			profiling_frame.clone(),
			Arc::new(AtomicBool::new(true)),
			Arc::new(AtomicU64::new(0)),
		)
	}

	/// Construction and destruction must run on a worker. The session token
	/// also stops playback if invalidation happens during device construction.
	#[profile_function("profiling_frame")]
	pub fn new_in_session(
		settings: Arc<RwLock<SpatialAudioSettings>>,
		content: Arc<SpatialAudioContent>,
		profiling_frame: Arc<ProfilingFrame>,
		session_active: Arc<AtomicBool>,
		object_playback_epoch: Arc<AtomicU64>,
	) -> Result<Self, SpatialAudioError> {
		debug!(
			settings = ?settings,
			"new()"
		);

		let profiling_frame_clone1 = profiling_frame.clone();
		let profiling_frame_clone2 = profiling_frame.clone();

		let settings_clone1 = settings.clone();
		let settings_clone2 = settings.clone();

		let content_clone = content.clone();

		let lifetime = PlaybackLifetime {
			running: Arc::new(AtomicBool::new(true)),
			session_active,
		};
		let depth_lifetime = lifetime.clone();
		let object_lifetime = lifetime.clone();

		let alto = Alto::load_default()?;
		let device = alto.open(None)?;
		if device.is_extension_present(Alc::SoftHrtf) {
			let attribs = ContextAttrs {
				soft_hrtf: Some(true),
				..Default::default()
			};
			device.soft_reset(Some(attribs)).unwrap();
		}
		let context = Arc::new(device.new_context(None)?);
		context.set_distance_model(DistanceModel::LinearClamped);
		context.set_position([0.0, 0.0, 0.0])?;
		let context_clone1 = context.clone();
		let context_clone2 = context.clone();

		let depth_audio_sources_data = Arc::new(RwLock::new(Vec::<DepthAudioSourceData>::new()));
		let depth_audio_sources_data_clone = depth_audio_sources_data.clone();

		let object_audio_sources_data = Arc::new(RwLock::new(ObjectAudioQueue::new(
			object_playback_epoch.clone(),
		)));
		let object_audio_sources_data_clone = object_audio_sources_data.clone();

		let mut audio = Self {
			_alto: alto,
			_device: device,
			_context: context,
			depth_audio_sources_data,
			depth_audio_thread: None,
			object_audio_thread: None,
			object_audio_sources_data,
			lifetime,
			settings,
			content,
			profiling_frame: profiling_frame_clone1,
		};
		// Build incrementally so a failed second spawn also stops/joins the first.
		audio.depth_audio_thread = Some(
			std::thread::Builder::new()
				.name("Depth Audio".to_string())
				.spawn(move || {
					depth_audio_thread(
						depth_lifetime,
						context_clone1,
						depth_audio_sources_data_clone,
						settings_clone1,
						&profiling_frame_clone2,
					)
				})?,
		);
		audio.object_audio_thread = Some(
			std::thread::Builder::new()
				.name("Object Audio".to_string())
				.spawn(move || {
					object_audio_thread(
						object_lifetime,
						settings_clone2,
						&content_clone.coco_labels_audio_file,
						context_clone2,
						object_audio_sources_data_clone,
					)
				})?,
		);
		Ok(audio)
	}

	/// Interrupts the current object announcement and invalidates every queued item.
	/// The token is owned by the audio session, so an old session cannot affect a new one.
	pub fn invalidate_object_audio_playback(&self) {
		self.object_audio_sources_data
			.read()
			.unwrap()
			.playback_epoch
			.fetch_add(1, Ordering::AcqRel);
		if let Some(thread) = &self.object_audio_thread {
			thread.thread().unpark();
		}
	}

	/// returns whether it needs to be recreated, as the output device changed
	pub fn update(
		&mut self,
		depth_estimation_data: &[f32; 256 * 256],
		object_detection_data: &[TrackedObject],
	) -> bool {
		if !self.lifetime.is_running() {
			return false;
		}
		let mut should_restart = false;

		#[allow(unused)]
		let profiling_scope = self.profiling_frame.scope("update");

		// this extension is implemented on android, such that we can restart SpatialAudio when the output device changes
		// on desktop (linux at least), this is not needed and the extension is not implemented, as output switching happens automatically
		if self._device.is_extension_present(Alc::Disconnect) {
			match self._device.connected() {
				Ok(connected) => {
					if !connected {
						error!("No audio device connected right now!");
						should_restart = true;
					}
				}
				Err(e) => {
					error!(
						"Failed to retrieve if device is connected, even though ALC_EXT_disconnect is present: {e}"
					);
				}
			}
		}

		let settings = self.settings.read().unwrap();
		let depth_audio_paused = settings.depth_audio_paused;
		let object_audio_paused = settings.object_audio_paused;

		if !depth_audio_paused {
			*self.depth_audio_sources_data.write().unwrap() = process_depth_estimation_data(
				depth_estimation_data,
				&settings,
				&self.profiling_frame,
			);
		}
		if object_detection_data.is_empty() {
			// A transient miss must not chop an announcement that has already started.
			// It only prevents stale pending announcements from starting afterwards.
			self.object_audio_sources_data
				.write()
				.unwrap()
				.update(VecDeque::new());
		} else if !object_audio_paused {
			let new_audio_sources_data = process_object_detection_data(
				depth_estimation_data,
				object_detection_data,
				&self.content.object_label_data,
				&self.profiling_frame,
			);
			self.object_audio_sources_data
				.write()
				.unwrap()
				.update(new_audio_sources_data);
		}

		should_restart
	}
}

fn depth_audio_thread(
	running: PlaybackLifetime,
	context: Arc<Context>,
	depth_audio_sources_data: Arc<RwLock<Vec<DepthAudioSourceData>>>,
	settings: Arc<RwLock<SpatialAudioSettings>>,
	profiling_frame: &ProfilingFrame,
) {
	if !running.is_running() {
		return;
	}
	debug!(
		sample_rate = SpatialAudioSettings::SAMPLE_RATE,
		buffer_duration = settings.read().unwrap().buffer_duration,
		"depth_audio_thread()"
	);

	let mut sources = (0..SpatialAudioSettings::NUMBER_OF_SOURCES)
		.map(|_| context.new_streaming_source().unwrap())
		.collect::<Vec<_>>();

	for source in sources.iter_mut() {
		source
			.set_max_distance(SpatialAudioSettings::MAX_DISTANCE)
			.unwrap();
		source
			.set_rolloff_factor(SpatialAudioSettings::ROLLOFF_FACTOR)
			.unwrap();
		source
			.set_reference_distance(SpatialAudioSettings::REFERENCE_DISTANCE)
			.unwrap();
		source.set_gain(0.5).unwrap();
	}

	let silent_audio_source_data = DepthAudioSourceData::new(
		0.0,
		settings.read().unwrap().buffer_duration,
		SpatialAudioSettings::SAMPLE_RATE,
		Vec3::default(),
		profiling_frame,
	);

	{
		let depth_audio_sources_data = depth_audio_sources_data.read().unwrap();

		for (i, source) in sources.iter_mut().enumerate() {
			let source_data = depth_audio_sources_data
				.get(i)
				.unwrap_or(&silent_audio_source_data);

			// fill all buffers with the same samples for now
			for _ in 0..SpatialAudioSettings::BUFFERS_PER_SOURCE {
				let buffer = context
					.new_buffer(&source_data.samples, source_data.sample_rate as i32)
					.unwrap();
				source.queue_buffer(buffer).unwrap();
			}
			source.set_position(source_data.position).unwrap();
			if running.is_running() {
				source.play();
			}
		}
	}

	while running.is_running() {
		if settings.read().unwrap().depth_audio_paused {
			std::thread::park_timeout(Duration::from_millis(20));
			continue;
		}

		{
			let depth_audio_sources_data = depth_audio_sources_data.read().unwrap();

			for (i, source) in sources.iter_mut().enumerate() {
				let source_data = depth_audio_sources_data
					.get(i)
					.unwrap_or(&silent_audio_source_data);

				if source.buffers_processed() > 0 {
					let mut unqueued_buffer = source.unqueue_buffer().unwrap();
					unqueued_buffer
						.set_data(&source_data.samples, source_data.sample_rate as i32)
						.unwrap();
					source.queue_buffer(unqueued_buffer).unwrap();
				}
				source.set_position(source_data.position).unwrap();
				source
					.set_max_distance(SpatialAudioSettings::MAX_DISTANCE)
					.unwrap();
				source
					.set_rolloff_factor(SpatialAudioSettings::ROLLOFF_FACTOR)
					.unwrap();
				source
					.set_reference_distance(SpatialAudioSettings::REFERENCE_DISTANCE)
					.unwrap();

				if running.is_running() && source.state() == SourceState::Stopped {
					source.play();
				}
			}
		}

		std::thread::park_timeout(Duration::from_millis(2));

		secondary_frame_mark!("Depth Audio Frame");
	}
}

fn object_audio_thread(
	running: PlaybackLifetime,
	settings: Arc<RwLock<SpatialAudioSettings>>,
	coco_audio_file: &AudioFileData,
	context: Arc<Context>,
	object_audio_sources_data: Arc<RwLock<ObjectAudioQueue>>,
) {
	if !running.is_running() {
		return;
	}
	let coco_audio_samples: &[i16] = &coco_audio_file.samples;

	debug!(
		sample_rate = coco_audio_file.sample_rate,
		"object_audio_thread()"
	);

	let mut source = context.new_static_source().unwrap();
	source.set_gain(0.5).unwrap();

	let mut sound_buffer: Vec<Mono<i16>> = Vec::new();

	while running.is_running() {
		if settings.read().unwrap().object_audio_paused {
			std::thread::park_timeout(Duration::from_millis(20));
			continue;
		}
		let Some((epoch, source_data)) = object_audio_sources_data.write().unwrap().pop() else {
			std::thread::park_timeout(Duration::from_millis(20));
			continue;
		};
		let sample_rate_ms = coco_audio_file.sample_rate as usize / 1000;
		let duration_ms = source_data.sound_end - source_data.sound_begin;
		sound_buffer.resize(sample_rate_ms * duration_ms, Mono::<i16> { center: 0 });
		let begin_sample = sample_rate_ms * source_data.sound_begin;
		let end_sample = sample_rate_ms * source_data.sound_end;
		sound_buffer.copy_from_slice(
			coco_audio_samples[begin_sample..end_sample]
				.iter()
				.map(|sample| Mono::<i16> { center: *sample })
				.collect::<Vec<Mono<i16>>>()
				.as_slice(),
		);
		let buffer = Arc::new(
			context
				.new_buffer(&sound_buffer, coco_audio_file.sample_rate as i32)
				.unwrap(),
		);
		if !running.is_running()
			|| object_audio_sources_data
				.read()
				.unwrap()
				.playback_epoch
				.load(Ordering::Acquire)
				!= epoch
		{
			continue;
		}
		source.set_buffer(buffer).unwrap();
		source.set_position(source_data.position).unwrap();
		source.play();

		while source.state() == SourceState::Playing {
			if !running.is_running()
				|| object_audio_sources_data
					.read()
					.unwrap()
					.playback_epoch
					.load(Ordering::Acquire)
					!= epoch
			{
				break;
			}
			std::thread::park_timeout(Duration::from_millis(20));
		}

		source.stop();
		source.clear_buffer();

		secondary_frame_mark!("Object Audio Frame");
	}
}

#[profile_function("profiling_frame")]
fn process_depth_estimation_data(
	depth_estimation_data: &[f32; 256 * 256],
	settings: &SpatialAudioSettings,
	profiling_frame: &ProfilingFrame,
) -> Vec<DepthAudioSourceData> {
	let step_size = (SpatialAudioSettings::PICTURE_RESOLUTION.x as f32
		/ (SpatialAudioSettings::NUMBER_OF_SOURCES as f32 - 1.0)) as usize;
	let mut audio_source_data = Vec::with_capacity(
		(SpatialAudioSettings::PICTURE_RESOLUTION.x as f32 / step_size as f32) as usize,
	);
	let mut calculate_sound_origin = CalculateSoundOrigin::new();

	let mut i: i32 = 0;
	while i < SpatialAudioSettings::PICTURE_RESOLUTION.x {
		let mut nearest_distance = f32::MAX;
		for j in 0..SpatialAudioSettings::PICTURE_RESOLUTION.y {
			let current_value = depth_estimation_data
				[(i + (j * SpatialAudioSettings::PICTURE_RESOLUTION.x)) as usize];
			nearest_distance = current_value.min(nearest_distance);
		}

		let sound_origin = calculate_sound_origin
			.calculate_sound_origin(IVec2 { x: i + 1, y: 0 }, nearest_distance);

		audio_source_data.push(DepthAudioSourceData::new(
			settings.frequency,
			settings.buffer_duration,
			SpatialAudioSettings::SAMPLE_RATE,
			sound_origin,
			profiling_frame,
		));

		// TODO: Why was that here? see old c++ code!
		if i == 0 {
			i -= 1;
		}

		i += step_size as i32;
	}

	audio_source_data
}

/// Converts a normalized detection coordinate to a depth-map pixel only when
/// it is a finite value inside the image. Kalman prediction is allowed to
/// leave the visible image; such a prediction must not be projected onto the
/// nearest edge and assigned an unrelated depth sample.
fn normalized_depth_coordinate(value: f32, resolution: i32) -> Option<i32> {
	if !value.is_finite() || !(0.0..=1.0).contains(&value) {
		return None;
	}

	let coordinate = (value * (resolution as f32 - 1.0)) as i32;
	(0..resolution).contains(&coordinate).then_some(coordinate)
}

fn depth_lookup_coordinate(center_x: f32, center_y: f32) -> Option<IVec2> {
	Some(IVec2 {
		x: normalized_depth_coordinate(center_x, SpatialAudioSettings::PICTURE_RESOLUTION.x)?,
		y: normalized_depth_coordinate(center_y, SpatialAudioSettings::PICTURE_RESOLUTION.y)?,
	})
}

#[profile_function("profiling_frame")]
fn process_object_detection_data(
	depth_estimation_data: &[f32; 256 * 256],
	object_detection_data: &[TrackedObject],
	object_label_data: &HashMap<String, ObjectLabelData>,
	profiling_frame: &ProfilingFrame,
) -> VecDeque<ObjectAudioSourceData> {
	let mut audio_source_data = VecDeque::new();

	for tracked_object in object_detection_data {
		let object = &tracked_object.object;

		let object_name = object.class_name.to_lowercase().trim().to_owned();

		let Some(object_label_data) = object_label_data.get(&object_name) else {
			error!(
				"[ProcessObjectDetectionData] Could not find object {} in the object_label_data. Skipping to next one ...",
				object_name
			);
			continue;
		};

		let Some(coord) = depth_lookup_coordinate(object.bbox.center_x, object.bbox.center_y)
		else {
			// A predicted track can be just outside the image or non-finite. It
			// remains valid tracker output, but has no safe audio/depth sample.
			continue;
		};

		trace!(
			"Object {}: Start: {} End: {}",
			object_name, object_label_data.sample_begin, object_label_data.sample_end
		);

		let depth_index = (coord.y as usize)
			.saturating_mul(SpatialAudioSettings::PICTURE_RESOLUTION.x as usize)
			.saturating_add(coord.x as usize);
		let Some(distance) = depth_estimation_data.get(depth_index).copied() else {
			// Keep this consumer defensive if the map resolution and its backing
			// representation ever diverge.
			continue;
		};
		let mut calculate_sound_origin = CalculateSoundOrigin::new();
		let sound_origin = calculate_sound_origin.calculate_sound_origin(coord, distance);

		audio_source_data.push_back(ObjectAudioSourceData {
			// TODO: object.tracking_id needs to be provided by object tracking, for now class id works as well
			object_id: object.class_id,
			name: object_name,
			sound_begin: object_label_data.sample_begin,
			sound_end: object_label_data.sample_end,
			position: sound_origin,
		});
	}

	audio_source_data
}

#[cfg(test)]
mod freshness_tests {
	use super::*;
	use std::sync::atomic::AtomicU64;

	fn queue() -> (Arc<AtomicU64>, ObjectAudioQueue) {
		let epoch = Arc::new(AtomicU64::new(0));
		(epoch.clone(), ObjectAudioQueue::new(epoch))
	}

	fn item(id: usize, x: f32) -> ObjectAudioSourceData {
		ObjectAudioSourceData {
			object_id: id,
			name: "person".into(),
			sound_begin: 0,
			sound_end: 1,
			position: Vec3 { x, y: 0.0, z: 1.0 },
		}
	}

	#[test]
	fn empty_snapshot_clears_pending_without_interrupting_popped_playback() {
		let (playback_epoch, mut queue) = queue();
		queue.update(VecDeque::from([item(1, 1.0), item(2, 2.0)]));
		let (playing_epoch, _) = queue.pop().unwrap();
		queue.update(VecDeque::new());
		assert!(queue.pop().is_none());
		assert_eq!(playback_epoch.load(Ordering::Acquire), playing_epoch);
	}

	#[test]
	fn explicit_invalidation_stops_popped_playback_and_discards_pending_items() {
		let (playback_epoch, mut queue) = queue();
		queue.update(VecDeque::from([item(1, 1.0), item(2, 2.0)]));
		let (playing_epoch, _) = queue.pop().unwrap();
		playback_epoch.fetch_add(1, Ordering::AcqRel);
		assert!(queue.pop().is_none());
		assert_ne!(playback_epoch.load(Ordering::Acquire), playing_epoch);

		// Reusing the same class/track ID after invalidation belongs to the new epoch.
		queue.update(VecDeque::from([item(1, 3.0)]));
		let (new_epoch, source) = queue.pop().unwrap();
		assert_ne!(new_epoch, playing_epoch);
		assert_eq!(source.position.x, 3.0);
	}

	#[test]
	fn fresh_updates_preserve_order_refresh_positions_and_remove_missing_objects() {
		let (_, mut queue) = queue();
		queue.update(VecDeque::from([item(1, 1.0), item(2, 2.0)]));
		queue.update(VecDeque::from([item(2, 4.0), item(3, 3.0)]));
		let (epoch, first) = queue.pop().unwrap();
		assert_eq!(epoch, 0);
		assert_eq!(first.object_id, 2);
		assert_eq!(first.position.x, 4.0);
		assert_eq!(queue.pop().unwrap().1.object_id, 3);
		assert!(queue.pop().is_none());
	}

	#[test]
	fn queue_keeps_existing_six_source_bound() {
		let (_, mut queue) = queue();
		queue.update((0..10).map(|id| item(id, 0.0)).collect());
		assert_eq!(queue.pending.len(), 6);
		assert_eq!(queue.pop().unwrap().1.object_id, 4);
	}
}

#[cfg(test)]
mod bounds_tests {
	use super::*;
	use crate::{BoundingBox, DetectedObject, TrackedObject};
	use std::{collections::HashMap, panic::AssertUnwindSafe};

	fn depth_map() -> [f32; 256 * 256] {
		[2.0; 256 * 256]
	}

	fn labels() -> HashMap<String, ObjectLabelData> {
		HashMap::from([(
			"person".to_string(),
			ObjectLabelData {
				sample_begin: 0,
				sample_end: 1,
			},
		)])
	}

	fn tracked_object(center_x: f32, center_y: f32, tracking_id: i32) -> TrackedObject {
		TrackedObject::new(
			DetectedObject::new(
				"person".to_string(),
				tracking_id.max(0) as usize,
				0.9,
				BoundingBox::new(center_x, center_y, 0.2, 0.2),
			),
			tracking_id,
		)
	}

	fn process(objects: &[TrackedObject]) -> std::thread::Result<VecDeque<ObjectAudioSourceData>> {
		let map = depth_map();
		let label_data = labels();
		let profiling_frame = ProfilingFrame::new("bounds-test");
		std::panic::catch_unwind(AssertUnwindSafe(|| {
			process_object_detection_data(&map, objects, &label_data, &profiling_frame)
		}))
	}

	#[test]
	fn normalized_coordinates_accept_only_finite_image_values() {
		for value in [0.0, 1.0] {
			assert!(normalized_depth_coordinate(value, 256).is_some());
		}
		for value in [
			-0.001,
			-1.0,
			1.001,
			2.0,
			f32::NAN,
			f32::INFINITY,
			f32::NEG_INFINITY,
		] {
			assert_eq!(
				normalized_depth_coordinate(value, 256),
				None,
				"value={value}"
			);
		}
	}

	#[test]
	fn valid_center_and_all_image_edges_keep_existing_audio_mapping() {
		let cases = [(0.5, 0.5), (0.0, 0.5), (0.5, 0.0), (1.0, 0.5), (0.5, 1.0)];

		for (index, (center_x, center_y)) in cases.into_iter().enumerate() {
			let output = process(&[tracked_object(center_x, center_y, index as i32)])
				.expect("valid coordinates must not panic");
			assert_eq!(output.len(), 1, "center=({center_x}, {center_y})");
			assert!(output[0].position.x.is_finite());
			assert!(output[0].position.y.is_finite());
		}
	}

	#[test]
	fn invalid_coordinates_are_discarded_without_affecting_valid_objects() {
		let invalid = [
			(-0.01, 0.5),
			(0.5, -0.01),
			(1.01, 0.5),
			(0.5, 1.01),
			(f32::NAN, 0.5),
			(0.5, f32::NAN),
			(f32::INFINITY, 0.5),
			(0.5, f32::INFINITY),
			(f32::NEG_INFINITY, 0.5),
			(0.5, f32::NEG_INFINITY),
		];

		for (index, (center_x, center_y)) in invalid.into_iter().enumerate() {
			let output = process(&[tracked_object(center_x, center_y, index as i32)])
				.expect("invalid coordinates must not panic");
			assert!(output.is_empty(), "center=({center_x}, {center_y})");
		}

		let output = process(&[
			tracked_object(-0.1, 0.5, 1),
			tracked_object(0.5, 0.5, 2),
			tracked_object(f32::INFINITY, 0.5, 3),
			tracked_object(1.0, 1.0, 4),
		])
		.expect("mixed coordinates must not panic");
		assert_eq!(output.len(), 2);
		assert_eq!(
			output
				.iter()
				.map(|source| source.object_id)
				.collect::<Vec<_>>(),
			vec![2, 4],
		);
	}
}
