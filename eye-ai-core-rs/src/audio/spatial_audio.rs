use alto::{
	Alto, Context, ContextAttrs, DeviceObject, DistanceModel, Mono, OutputDevice, Source,
	SourceState, ext::Alc,
};
use eye_ai_core_rs_profiling_attribute::profile_function;
use std::{
	collections::{HashMap, VecDeque},
	sync::{
		Arc, RwLock,
		atomic::{AtomicBool, Ordering},
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
}

/// Empty input invalidates pending AND currently playing object positions. The playback
/// epoch is taken with the item under the same lock; a clear cannot make an old item fresh.
#[derive(Default)]
struct ObjectAudioQueue {
	pending: VecDeque<ObjectAudioSourceData>,
	epoch: u64,
}

impl ObjectAudioQueue {
	fn update(&mut self, sources: VecDeque<ObjectAudioSourceData>) {
		if sources.is_empty() {
			self.pending.clear();
			self.epoch = self.epoch.wrapping_add(1);
			return;
		}
		self.pending
			.retain(|old| sources.iter().any(|new| old.object_id == new.object_id));
		for source in sources {
			if let Some(old) = self
				.pending
				.iter_mut()
				.find(|old| old.object_id == source.object_id)
			{
				*old = source;
			} else {
				while self.pending.len() >= 6 {
					self.pending.pop_front();
				}
				self.pending.push_back(source);
			}
		}
	}

	fn pop(&mut self) -> Option<(u64, ObjectAudioSourceData)> {
		self.pending.pop_front().map(|source| (self.epoch, source))
	}
}

pub struct SpatialAudio {
	_alto: Alto,
	_device: OutputDevice,
	_context: Arc<Context>,
	depth_audio_sources_data: Arc<RwLock<Vec<DepthAudioSourceData>>>,
	_depth_audio_thread: std::thread::JoinHandle<()>,
	depth_audio_running: Arc<AtomicBool>,
	object_audio_sources_data: Arc<RwLock<ObjectAudioQueue>>,
	_object_audio_thread: std::thread::JoinHandle<()>,
	object_audio_running: Arc<AtomicBool>,
	pub settings: Arc<RwLock<SpatialAudioSettings>>,
	content: Arc<SpatialAudioContent>,
	profiling_frame: Arc<ProfilingFrame>,
}
impl Drop for SpatialAudio {
	fn drop(&mut self) {
		self.depth_audio_running.store(false, Ordering::Relaxed);
		self.object_audio_running.store(false, Ordering::Relaxed);
	}
}
impl SpatialAudio {
	#[profile_function("profiling_frame")]
	pub fn new(
		settings: Arc<RwLock<SpatialAudioSettings>>,
		content: Arc<SpatialAudioContent>,
		profiling_frame: Arc<ProfilingFrame>,
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

		let depth_audio_running = Arc::new(AtomicBool::new(true));
		let depth_audio_running_clone = depth_audio_running.clone();

		let object_audio_running = Arc::new(AtomicBool::new(true));
		let object_audio_running_clone = depth_audio_running.clone();

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

		let object_audio_sources_data = Arc::new(RwLock::new(ObjectAudioQueue::default()));
		let object_audio_sources_data_clone = object_audio_sources_data.clone();

		Ok(Self {
			_alto: alto,
			_device: device,
			_context: context,
			depth_audio_sources_data,
			_depth_audio_thread: std::thread::Builder::new()
				.name("Depth Audio".to_string())
				.spawn(move || {
					depth_audio_thread(
						depth_audio_running_clone,
						context_clone1,
						depth_audio_sources_data_clone,
						settings_clone1,
						&profiling_frame_clone2,
					)
				})
				.expect("failed to spawn depth audio thread"),
			depth_audio_running,
			object_audio_sources_data,
			_object_audio_thread: std::thread::Builder::new()
				.name("Object Audio".to_string())
				.spawn(move || {
					object_audio_thread(
						object_audio_running_clone,
						settings_clone2,
						&content_clone.coco_labels_audio_file,
						context_clone2,
						object_audio_sources_data_clone,
					)
				})
				.expect("failed to spawn object audio thread"),
			object_audio_running,
			settings,
			content,
			profiling_frame: profiling_frame_clone1,
		})
	}

	/// returns whether it needs to be recreated, as the output device changed
	pub fn update(
		&mut self,
		depth_estimation_data: &[f32; 256 * 256],
		object_detection_data: &[TrackedObject],
	) -> bool {
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
			// Invalidation must also reach the queue while user/TTS pause is active.
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
	running: Arc<AtomicBool>,
	context: Arc<Context>,
	depth_audio_sources_data: Arc<RwLock<Vec<DepthAudioSourceData>>>,
	settings: Arc<RwLock<SpatialAudioSettings>>,
	profiling_frame: &ProfilingFrame,
) {
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
			source.play();
		}
	}

	while running.load(Ordering::Relaxed) {
		if settings.read().unwrap().depth_audio_paused {
			std::thread::sleep(Duration::from_millis(500));
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

				if source.state() == SourceState::Stopped {
					source.play();
				}
			}
		}

		std::thread::sleep(Duration::from_millis(2));

		secondary_frame_mark!("Depth Audio Frame");
	}
}

fn object_audio_thread(
	running: Arc<AtomicBool>,
	settings: Arc<RwLock<SpatialAudioSettings>>,
	coco_audio_file: &AudioFileData,
	context: Arc<Context>,
	object_audio_sources_data: Arc<RwLock<ObjectAudioQueue>>,
) {
	let coco_audio_samples: &[i16] = &coco_audio_file.samples;

	debug!(
		sample_rate = coco_audio_file.sample_rate,
		"object_audio_thread()"
	);

	let mut source = context.new_static_source().unwrap();
	source.set_gain(0.5).unwrap();

	let mut sound_buffer: Vec<Mono<i16>> = Vec::new();

	while running.load(Ordering::Relaxed) {
		if settings.read().unwrap().object_audio_paused {
			std::thread::sleep(Duration::from_millis(500));
			continue;
		}
		let Some((epoch, source_data)) = object_audio_sources_data.write().unwrap().pop() else {
			std::thread::sleep(Duration::from_millis(250));
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
		if object_audio_sources_data.read().unwrap().epoch != epoch {
			continue;
		}
		source.set_buffer(buffer).unwrap();
		source.set_position(source_data.position).unwrap();
		source.play();

		while source.state() == SourceState::Playing {
			if !running.load(Ordering::Relaxed)
				|| object_audio_sources_data.read().unwrap().epoch != epoch
			{
				break;
			}
			std::thread::sleep(Duration::from_millis(100));
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

		let coord = IVec2 {
			x: (object.bbox.center_x * (SpatialAudioSettings::PICTURE_RESOLUTION.x as f32 - 1.0))
				as i32,
			y: (object.bbox.center_y * (SpatialAudioSettings::PICTURE_RESOLUTION.y as f32 - 1.0))
				as i32,
		};

		trace!(
			"Object {}: Start: {} End: {}",
			object_name, object_label_data.sample_begin, object_label_data.sample_end
		);

		let distance = depth_estimation_data
			[(coord.x + (coord.y * SpatialAudioSettings::PICTURE_RESOLUTION.x)) as usize];
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
	fn empty_snapshot_invalidates_pending_and_popped_playback() {
		let mut queue = ObjectAudioQueue::default();
		queue.update(VecDeque::from([item(1, 1.0), item(2, 2.0)]));
		let (playing_epoch, _) = queue.pop().unwrap();
		queue.update(VecDeque::new());
		assert!(queue.pop().is_none());
		assert_ne!(queue.epoch, playing_epoch);
		// Reusing the same class/track ID in another generation cannot revive the old clip.
		queue.update(VecDeque::from([item(1, 3.0)]));
		let (new_epoch, source) = queue.pop().unwrap();
		assert_ne!(new_epoch, playing_epoch);
		assert_eq!(source.position.x, 3.0);
	}

	#[test]
	fn fresh_updates_preserve_order_refresh_positions_and_remove_missing_objects() {
		let mut queue = ObjectAudioQueue::default();
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
		let mut queue = ObjectAudioQueue::default();
		queue.update((0..10).map(|id| item(id, 0.0)).collect());
		assert_eq!(queue.pending.len(), 6);
		assert_eq!(queue.pop().unwrap().1.object_id, 4);
	}
}
