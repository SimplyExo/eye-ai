use alto::{
	Alto, Context, ContextAttrs, DeviceObject, DistanceModel, Mono, OutputDevice, Source,
	SourceState, ext::Alc,
};
use eye_ai_core_rs_profiling_attribute::profile_function;
use json::JsonValue;
use std::{
	collections::{HashMap, VecDeque},
	sync::{
		Arc, RwLock,
		atomic::{AtomicBool, Ordering},
	},
	time::Duration,
};
use thiserror::Error;
use tracing_tracy::client::secondary_frame_mark;

use crate::{
	ProfilingFrame, TrackedObject,
	audio::{
		CalculateSoundOrigin, DepthAudioSourceData, IVec2, ObjectAudioSourceData,
		SpatialAudioSettings, Vec3, read_audio_file,
	},
};

pub type AudioLogCallback = Arc<dyn Fn(&str)>;

#[derive(Debug, Error)]
pub enum SpatialAudioError {
	#[error("Alto error: {0}")]
	AltoError(#[from] alto::AltoError),
	#[error("JSON error: {0}")]
	JsonError(#[from] json::Error),
}

struct ObjectLabelData {
	sample_begin: usize,
	sample_end: usize,
}
#[profile_function("profiling_frame")]
fn read_object_label_data(
	json_content: &str,
	profiling_frame: &ProfilingFrame,
) -> Result<HashMap<String, ObjectLabelData>, json::Error> {
	let json = json::parse(json_content)?;
	let JsonValue::Array(json_array) = json else {
		return Err(json::Error::WrongType("json should be array!".to_string()));
	};
	let mut map = HashMap::<String, ObjectLabelData>::new();
	for json_data in json_array {
		let JsonValue::Object(data_object) = json_data else {
			return Err(json::Error::wrong_type("array element should be object!"));
		};
		let object_label_data = ObjectLabelData {
			sample_begin: data_object["start"]
				.as_usize()
				.ok_or(json::Error::wrong_type("usize"))?,
			sample_end: data_object["end"]
				.as_usize()
				.ok_or(json::Error::wrong_type("usize"))?,
		};
		let label = data_object["text"]
			.as_str()
			.ok_or(json::Error::wrong_type("string"))?;
		map.insert(label.to_string().to_lowercase(), object_label_data);
	}
	Ok(map)
}

pub struct SpatialAudio {
	_alto: Alto,
	_device: OutputDevice,
	_context: Arc<Context>,
	depth_audio_sources_data: Arc<RwLock<Vec<DepthAudioSourceData>>>,
	_depth_audio_thread: std::thread::JoinHandle<()>,
	depth_audio_running: Arc<AtomicBool>,
	object_audio_sources_data: Arc<RwLock<VecDeque<ObjectAudioSourceData>>>,
	_object_audio_thread: std::thread::JoinHandle<()>,
	object_audio_running: Arc<AtomicBool>,
	pub settings: Arc<RwLock<SpatialAudioSettings>>,
	object_label_data: HashMap<String, ObjectLabelData>,
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
		settings: SpatialAudioSettings,
		profiling_frame: Arc<ProfilingFrame>,
	) -> Result<Self, SpatialAudioError> {
		let profiling_frame_clone1 = profiling_frame.clone();
		let profiling_frame_clone2 = profiling_frame.clone();
		let profiling_frame_clone3 = profiling_frame.clone();

		let object_label_data =
			read_object_label_data(&settings.coco_labels_json_content, &profiling_frame)?;

		let settings = Arc::new(RwLock::new(settings));
		let settings_clone1 = settings.clone();
		let settings_clone2 = settings.clone();

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

		let object_audio_sources_data =
			Arc::new(RwLock::new(VecDeque::<ObjectAudioSourceData>::new()));
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
						context_clone2,
						object_audio_sources_data_clone,
						&profiling_frame_clone3,
					)
				})
				.expect("failed to spawn object audio thread"),
			object_audio_running,
			settings,
			object_label_data,
			profiling_frame: profiling_frame_clone1,
		})
	}

	#[profile_function("self.profiling_frame")]
	pub fn update(
		&mut self,
		depth_estimation_data: &[f32; 256 * 256],
		object_detection_data: &[TrackedObject],
		log_info_callback: AudioLogCallback,
		log_error_callback: AudioLogCallback,
	) {
		let settings = self.settings.read().unwrap();
		if !settings.depth_audio_paused {
			*self.depth_audio_sources_data.write().unwrap() = process_depth_estimation_data(
				depth_estimation_data,
				&settings,
				&self.profiling_frame,
			);
		}
		if !settings.object_audio_paused {
			let new_audio_sources_data = process_object_detection_data(
				depth_estimation_data,
				object_detection_data,
				&self.object_label_data,
				&self.profiling_frame,
				log_info_callback,
				log_error_callback,
			);
			let mut object_audio_sources_data = self.object_audio_sources_data.write().unwrap();
			for new_source_data in new_audio_sources_data {
				let mut found = false;
				for source_data in object_audio_sources_data.iter_mut() {
					if source_data.object_id == new_source_data.object_id {
						found = true;
						*source_data = new_source_data.clone();
						break;
					}
				}
				if !found {
					// max of 6 sources at once
					while object_audio_sources_data.len() >= 6 {
						object_audio_sources_data.pop_front();
					}
					object_audio_sources_data.push_back(new_source_data);
				}
			}
		}
	}
}

#[profile_function("profiling_frame")]
fn depth_audio_thread(
	running: Arc<AtomicBool>,
	context: Arc<Context>,
	depth_audio_sources_data: Arc<RwLock<Vec<DepthAudioSourceData>>>,
	settings: Arc<RwLock<SpatialAudioSettings>>,
	profiling_frame: &ProfilingFrame,
) {
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

#[profile_function("profiling_frame")]
fn object_audio_thread(
	running: Arc<AtomicBool>,
	settings: Arc<RwLock<SpatialAudioSettings>>,
	context: Arc<Context>,
	object_audio_sources_data: Arc<RwLock<VecDeque<ObjectAudioSourceData>>>,
	profiling_frame: &ProfilingFrame,
) {
	let audio_file_content = settings
		.read()
		.unwrap()
		.coco_labels_audio_file_content
		.clone()
		.into_boxed_slice();

	let audio_file_data = read_audio_file(audio_file_content, profiling_frame)
		.expect("failed to load coco labels audio file");

	let coco_audio_samples: Vec<i16> = audio_file_data.samples;
	/*let info_callback = settings.read().unwrap().log_info_callback;
	(info_callback)(
		format!(
			"[LoadAudioLabelsFile] File sample rate: {}",
			audio_file_data.sample_rate
		)
		.as_str(),
	);*/

	let mut source = context.new_static_source().unwrap();
	source.set_gain(0.5).unwrap();

	let mut sound_buffer: Vec<Mono<i16>> = Vec::new();

	while running.load(Ordering::Relaxed) {
		if settings.read().unwrap().object_audio_paused {
			std::thread::sleep(Duration::from_millis(500));
			continue;
		}
		let Some(source_data) = object_audio_sources_data.write().unwrap().pop_front() else {
			std::thread::sleep(Duration::from_millis(250));
			continue;
		};
		let sample_rate_ms = audio_file_data.sample_rate as usize / 1000;
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
				.new_buffer(&sound_buffer, audio_file_data.sample_rate as i32)
				.unwrap(),
		);
		source.set_buffer(buffer).unwrap();
		source.set_position(source_data.position).unwrap();
		source.play();

		while source.state() == SourceState::Playing {
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
	let mut calculate_sound_origin = CalculateSoundOrigin::new(profiling_frame);

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
	log_info_callback: AudioLogCallback,
	log_error_callback: AudioLogCallback,
) -> VecDeque<ObjectAudioSourceData> {
	let mut audio_source_data = VecDeque::new();

	for tracked_object in object_detection_data {
		let object = &tracked_object.object;

		let object_name = object.class_name.to_lowercase().trim().to_owned();

		let Some(object_label_data) = object_label_data.get(&object_name) else {
			log_error_callback(format!(
					"[ProcessObjectDetectionData] Could not find object {} in the object_label_data. Skipping to next one ...",
					object_name
				).as_str());
			continue;
		};

		let coord = IVec2 {
			x: (object.bbox.center_x * (SpatialAudioSettings::PICTURE_RESOLUTION.x as f32 - 1.0))
				as i32,
			y: (object.bbox.center_y * (SpatialAudioSettings::PICTURE_RESOLUTION.y as f32 - 1.0))
				as i32,
		};

		log_info_callback(
			format!(
				"Object {}: Start: {} End: {}",
				object_name, object_label_data.sample_begin, object_label_data.sample_end
			)
			.as_str(),
		);

		let distance = depth_estimation_data
			[(coord.x + (coord.y * SpatialAudioSettings::PICTURE_RESOLUTION.x)) as usize];
		let mut calculate_sound_origin = CalculateSoundOrigin::new(profiling_frame);
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
