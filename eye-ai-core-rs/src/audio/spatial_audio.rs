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

struct ObjectAudioQueue {
	pending: VecDeque<QueuedObjectAudio>,
	playback_epoch: Arc<AtomicU64>,
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
#[path = "spatial_audio/tests/mod.rs"]
mod tests;
